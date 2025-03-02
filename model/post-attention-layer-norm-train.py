import argparse
parser = argparse.ArgumentParser(description="Compute empirical confidence for GSM8K using flash attention and post-attention layer norms")
parser.add_argument("--visible_cudas", type=str, help="CUDA devices to use")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--goal", type=str, required=True, help="Predictionn goal ('conf', 'true')")
parser.add_argument("--llm_id", type=str, required=True, help="ID of the LLM model")
parser.add_argument("--wconf_dir_path", type=str, required=True, help="Path to the directory containing the wconf files")
parser.add_argument("--postfix", type=str, required=True, help="Postfix used in input file names, e.g. '_wconf', '_wconf_wid', etc.")
parser.add_argument("--shots_dir_path", type=str, required=True, help="Path to the directory containing the few-shot examples")
parser.add_argument("--nshots", type=int, default=10, help="Number of shots in few-shot prompting")
parser.add_argument("--batch_q", type=int, default=30, help="number of unique questions per batch")
parser.add_argument("--nruns", type=int, default=30, help="Number of LLM runs per sample")
parser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length for the tokenizer")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility")
# New hyperparameters for ConfidenceHeadStackedInput:
parser.add_argument("--conv_channels", type=int, default=32, help="Number of channels for the convolution in the confidence head")
parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for the convolution in the confidence head")
parser.add_argument("--conv_activation", type=str, default="relu", help="Activation function for the convolution ('relu', 'gelu', etc.)")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the confidence head")
parser.add_argument("--output_dir", type=str, default="../storage/trained_models", help="Base directory to save training artifacts")
args = parser.parse_args()

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

# Import from utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (seed_everything, load_wconf_dataset, load_few_shot_examples, load_system_prompt,
                     flatten_dataset, assert_flatten_dataset, get_last_hidden_state_size,
                     initialize_tokenizer, create_collate_fn, setup_data_loaders, 
                     plot_all_metrics, compute_metrics, save_predictions_and_model)
from talk2llm import Talk2LLM
sys.path = original_sys_path
seed_everything(args.seed)
print(f"Random seed set to {args.seed}")

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from torch.optim import Adam
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM
from confhead import ConfidenceHeadStackedInput
import gc
import datetime


def evaluate_model(loader, transformer_model, conf_head, tokenizer, device, phase="Val", goal="conf"):
    """Evaluate the confidence head on a given loader.
       Returns: average loss, metrics dict, and predictions and targets."""
    conf_head.eval()
    all_preds = []
    all_targets = []
    epoch_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"Evaluating ({phase})")
    
    with torch.no_grad():
        for inputs, targets, texts, texts_trunc_pad in pbar:
            with torch.autocast('cuda', dtype=torch.float16):
                _ = transformer_model(**inputs)
            stacked_hidden = extract_eos_per_head(tokenizer, inputs, device).float()
            preds = conf_head(stacked_hidden)
            loss = nn.BCELoss()(preds, targets.float())
            epoch_loss += loss.item() * targets.size(0)
            count += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_description(f"Processed {phase} Batch")
            
            # Free memory after each batch.
            del inputs, targets, texts, texts_trunc_pad, stacked_hidden, preds, loss
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
    pbar.close()
    avg_loss = epoch_loss / count
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), goal=goal)
    return avg_loss, metrics, np.array(all_targets), np.array(all_preds)


# -------------------------------
# Setup output directories.

output_dir = os.path.join(args.output_dir, args.llm_id.replace("/", "-"), args.goal, f"conv{args.conv_channels}_k{args.kernel_size}_act{args.conv_activation}_drop{args.dropout}_dt{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
os.makedirs(output_dir, exist_ok=True)
epoch_dir = os.path.join(output_dir, "epoch_weights")
os.makedirs(epoch_dir, exist_ok=True)
best_dir = os.path.join(output_dir, "best")
os.makedirs(best_dir, exist_ok=True)
preds_dir = os.path.join(output_dir, "predictions")
os.makedirs(preds_dir, exist_ok=True)

# -------------------------------
# Load the model using flash_attention_2 with device_map="auto".
dtype = torch.float16
model_id = args.llm_id
tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=args.max_seq_length)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=dtype, 
    attn_implementation="flash_attention_2",
    device_map="auto"  # This will split the model across available GPUs.
)
model.eval()

# Register forward hooks on each decoder layer's post_attention_layernorm.
post_attn_outputs = {}
def make_hook(layer_idx):
    def hook_fn(module, input, output):
        post_attn_outputs[layer_idx] = output.detach()
    return hook_fn
for idx, layer in enumerate(model.model.layers):
    layer.post_attention_layernorm.register_forward_hook(make_hook(idx))

# -------------------------------
# Hyperparameters for ConfidenceHeadStackedInput.
num_heads = 4
head_dim = 1024
num_blocks = len(model.model.layers)  # e.g., 32
stacked_channels = num_blocks * num_heads

# -------------------------------
# Function to extract per-head EOS representations.
def extract_eos_per_head(tokenizer, inputs, desired_device):
    batch_hidden = []
    eos_token_id = tokenizer.eos_token_id
    input_ids = inputs["input_ids"]
    input_ids_list = input_ids.tolist()
    for idx in range(num_blocks):
        block_out = post_attn_outputs[idx].to(desired_device)
        block_eos = []
        for i, seq in enumerate(input_ids_list):
            # try:
            #     eos_index = seq.index(eos_token_id)
            # except ValueError:
            #     eos_index = len(seq) - 1
            eos_index = -1
            rep = block_out[i, eos_index, :]
            rep = rep.view(num_heads, head_dim)
            block_eos.append(rep)
        block_eos_tensor = torch.stack(block_eos, dim=0)
        batch_hidden.append(block_eos_tensor)
    stacked = torch.stack(batch_hidden, dim=1)
    stacked = stacked.view(stacked.size(0), -1, head_dim)
    return stacked.to(desired_device)

# -------------------------------
# Setup model and optimizer for ConfidenceHeadStackedInput.
def setup_model_and_optimizer(num_heads, head_dim, num_blocks, device, conv_channels, kernel_size, conv_activation, dropout):
    confidence_head = ConfidenceHeadStackedInput(num_heads=num_blocks*num_heads, head_dim=head_dim,
                                                 conv_channels=conv_channels, kernel_size=kernel_size,
                                                 conv_activation=conv_activation, dropout=dropout)
    confidence_head.to(device)
    optimizer = Adam(confidence_head.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    return confidence_head, optimizer, criterion

# -------------------------------
# Training loop functions.
def train_epoch(conf_head, transformer_model, train_loader, optimizer, criterion, tokenizer, device, use_tqdm=False):
    epoch_batch_losses = []
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc="Training Loss: N/A")
    global post_attn_outputs
    for batch_idx, (inputs, targets, texts, texts_trunc_pad) in enumerate(pbar):
        post_attn_outputs = {}
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                _ = transformer_model(**inputs)
        stacked_hidden = extract_eos_per_head(tokenizer, inputs, device).float()
        pred_conf = conf_head(stacked_hidden)
        loss = criterion(pred_conf, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_batch_losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

        # Manually free memory after each batch.
        del inputs, targets, texts, texts_trunc_pad, stacked_hidden, pred_conf, loss
        torch.cuda.empty_cache()
        gc.collect()

    pbar.close()
    return epoch_loss / len(train_loader), epoch_batch_losses

def train_model_main(conf_head, transformer_model, train_loader, val_loader, test_loader, optimizer, criterion, tokenizer, device, num_epochs=5, use_tqdm=False, goal="conf"):
    print(f"Training for {num_epochs} epochs over {len(train_loader.dataset)} samples ({len(train_loader)} batches per epoch).")
    epoch_losses, val_loss_history, test_loss_history, bce_history_train, brier_history_train, ece_history_train, ece_manual_history_train, bce_history_val, brier_history_val, ece_history_val, ece_manual_history_val, bce_history_test, brier_history_test, ece_history_test, ece_manual_history_test = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    acc_history_val, acc_history_test = [], []
    prec_history_val, prec_history_test = [], []
    rec_history_val, rec_history_test = [], []
    f1_history_val, f1_history_test = [], []
    aucroc_history_val, aucroc_history_test = [], []
    aucpr_history_val, aucpr_history_test = [], []
    best_val_loss = None
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1} =====")
        train_loss, batch_losses = train_epoch(conf_head, transformer_model, train_loader, optimizer, criterion, tokenizer, device, use_tqdm)
        epoch_losses.append(train_loss)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
        val_loss, val_metrics, val_targets, val_preds = evaluate_model(val_loader, transformer_model, conf_head, tokenizer, device, phase="Val", goal=goal)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        test_loss, test_metrics, test_targets, test_preds = evaluate_model(test_loader, transformer_model, conf_head, tokenizer, device, phase="Test", goal=goal)
        print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}, Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Save predictions and model weights
        save_predictions_and_model(preds_dir, epoch_dir, epoch, val_targets, val_preds, test_targets, test_preds, conf_head)
        # Save best model based on validation loss
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(conf_head.state_dict(), os.path.join(best_dir, "conf_head_best.pt"))
            with open(os.path.join(best_dir, "best_metrics.txt"), "w") as f:
                f.write(f"Epoch {epoch+1}\nValidation Loss: {val_loss:.4f}\nMetrics: {val_metrics}\n")
            print(f"Best model saved at epoch {epoch+1}.")
        
        bce_history_val.append(val_metrics["BCE Loss"])
        brier_history_val.append(val_metrics["Brier Score"])
        ece_history_val.append(val_metrics["ECE"])
        ece_manual_history_val.append(val_metrics["ECE Manual"])
        
        bce_history_test.append(test_metrics["BCE Loss"])
        brier_history_test.append(test_metrics["Brier Score"])
        ece_history_test.append(test_metrics["ECE"])
        ece_manual_history_test.append(test_metrics["ECE Manual"])
        
        val_loss_history.append(val_loss)
        test_loss_history.append(test_loss)

        if goal == "true":
            acc_history_val.append(val_metrics["accuracy"])
            acc_history_test.append(test_metrics["accuracy"])
            prec_history_val.append(val_metrics["precision"])
            prec_history_test.append(test_metrics["precision"])
            rec_history_val.append(val_metrics["recall"])
            rec_history_test.append(test_metrics["recall"])
            f1_history_val.append(val_metrics["f1"])
            f1_history_test.append(test_metrics["f1"])
            aucroc_history_val.append(val_metrics["aucroc"])
            aucroc_history_test.append(test_metrics["aucroc"])
            aucpr_history_val.append(val_metrics["aucpr"])
            aucpr_history_test.append(test_metrics["aucpr"])

    print("Training complete.")
    metrics_history = {
        "train_loss": epoch_losses,
        "val_loss": val_loss_history,
        "test_loss": test_loss_history,
        "BCE": {"train": bce_history_train, "val": bce_history_val, "test": bce_history_test},
        "Brier": {"val": brier_history_val, "test": brier_history_test},
        "ECE": {"val": ece_history_val, "test": ece_history_test},
        "ECE Manual": {"val": ece_manual_history_val, "test": ece_manual_history_test},
    }
    if goal == "true":
        metrics_history.update({
            "accuracy": {"val": acc_history_val, "test": acc_history_test},
            "precision": {"val": prec_history_val, "test": prec_history_test},
            "recall": {"val": rec_history_val, "test": rec_history_test},
            "f1": {"val": f1_history_val, "test": f1_history_test},
            "AUC ROC": {"val": aucroc_history_val, "test": aucroc_history_test},
            "AUC PR": {"val": aucpr_history_val, "test": aucpr_history_test},
        })
    return metrics_history

# -------------------------------
# Main function.
def main(dataset_name, goal, llm_id, wconf_dir_path, postfix, shots_dir_path, nshots, max_seq_length, epochs, batch_q, nruns, seed, conv_channels, kernel_size, conv_activation, dropout):
    dataset = load_wconf_dataset(wconf_dir_path, postfix)
    # dataset['train'] = dataset['train'][:1]
    # dataset['val'] = dataset['val'][:1]
    # dataset['test'] = dataset['test'][:1]
    tokenizer = initialize_tokenizer(llm_id, max_seq_length)
    few_shot_prompt = load_few_shot_examples(shots_dir_path, nshots)
    system_prompt = load_system_prompt(dataset_name)
    dataset_flat = flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer, goal=goal)
    # For debugging, you might trim splits here if desired.
    assert_flatten_dataset(dataset_flat, tokenizer)
    
    transformer_model = model  # already loaded (model parallelism via device_map="auto")
    hidden_size = get_last_hidden_state_size(llm_id)  # expected to be 4096
    device = "cuda:0"
    print(f"Using device: {device} for training the confidence head.")
    
    num_heads = 4
    head_dim = 1024
    num_blocks = len(transformer_model.model.layers)
    
    conf_head, optimizer, criterion = setup_model_and_optimizer(num_heads, head_dim, num_blocks, device,
                                                                  conv_channels=conv_channels,
                                                                  kernel_size=kernel_size,
                                                                  conv_activation=conv_activation,
                                                                  dropout=dropout)
    
    flattened_batch_size = batch_q * nruns
    collate_fn = create_collate_fn(tokenizer, device, max_seq_length, goal=goal)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset_flat, flattened_batch_size, collate_fn)
    
    metrics_history = train_model_main(conf_head, transformer_model, train_loader, val_loader, test_loader,
                                    optimizer, criterion, tokenizer, device, num_epochs=epochs, use_tqdm=False, goal=goal)
    
    # Save training loss plot and additional metrics plots.
    plot_all_metrics(metrics_history, output_dir, epochs, goal=goal)
    print('=' * 50)
    print(f"✅ Done! Training artifacts saved to: {output_dir}")
    print('=' * 50)

if __name__ == "__main__":
    main(dataset_name=args.dataset_name, 
         goal=args.goal,
         llm_id=args.llm_id, 
         wconf_dir_path=args.wconf_dir_path, 
         postfix=args.postfix,
         shots_dir_path=args.shots_dir_path,
         nshots=args.nshots, 
         max_seq_length=args.max_seq_length, 
         epochs=args.epochs, 
         batch_q=args.batch_q, 
         nruns=args.nruns, 
         seed=args.seed,
         conv_channels=args.conv_channels,
         kernel_size=args.kernel_size,
         conv_activation=args.conv_activation,
         dropout=args.dropout)


# python post-attention-layer-norm-train.py --visible_cudas "0,1,2" --dataset_name "GSM8K" --goal "true" --llm_id "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" --wconf_dir_path "../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/subset" --postfix "_wconf_wid_sub" --shots_dir_path "../data/GSM8K/" --nshots 10 --max_seq_length 1024 --epochs 5 --batch_q 1 --nruns 30 --seed 23 --conv_channels 32 --kernel_size 1 --conv_activation "relu" --dropout 0.1