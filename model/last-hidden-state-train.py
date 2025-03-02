import argparse
parser = argparse.ArgumentParser(description="Compute empirical confidence for GSM8K using vLLM")
parser.add_argument("--visible_cudas", type=str, help="CUDA devices to use")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--llm_id", type=str, required=True, help="ID of the LLM model")
parser.add_argument("--gpu_memory", type=float, default=0.5, help="GPU memory utilization")
parser.add_argument("--tensor_parallel", type=int, default=2, help="Tensor parallel size")
parser.add_argument("--wconf_dir_path", type=str, required=True, help="Path to the directory containing the wconf files")
parser.add_argument("--nshots", type=int, default=10, help="Number of shots in few-shot prompting")
parser.add_argument("--batch_q", type=int, default=30, help="number of unique questions per batch")
parser.add_argument("--nruns", type=int, default=30, help="Number of LLM runs per sample")
parser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length for the tokenizer")
parser.add_argument("--vllm_task", type=str, default="reward", help="Task for vLLM")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility")
args = parser.parse_args()

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

# Import from utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import load_wconf_dataset, load_few_shot_examples, load_system_prompt, flatten_dataset, assert_flatten_dataset, get_last_hidden_state_size, initialize_tokenizer, create_collate_fn, setup_data_loaders, save_training_artifacts
from talk2llm import Talk2LLM
sys.path = original_sys_path

import torch
from torch import nn
from transformers import AutoTokenizer
import numpy as np
from torch.optim import Adam
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM
from confhead import ConfidenceHead


def setup_model_and_optimizer(hidden_size, device):
    confidence_head = ConfidenceHead(hidden_size)
    confidence_head.to(device)
    optimizer = Adam(confidence_head.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    return confidence_head, optimizer, criterion


def extract_eos_hidden_states(outputs, texts, inputs, tokenizer, device):
    """
    Extract hidden states from the EOS tokens, falling back to last token if no EOS found.
    """
    all_hidden_states = [output.outputs.data for output in outputs]
    
    eos_hidden_states = []
    for i, text in enumerate(texts):
        eos_token_id = tokenizer.eos_token_id
        input_ids = inputs["input_ids"][i].tolist()
        
        try:
            # Try to find EOS token
            eos_index = input_ids.index(eos_token_id)
        except ValueError:
            # If no EOS token found, use the last token
            eos_index = len(input_ids) - 1
            
        eos_hidden_states.append(all_hidden_states[i][eos_index])
    
    return torch.stack(eos_hidden_states).to(device)


def train_epoch(model, llm, train_loader, optimizer, criterion, tokenizer, device, use_tqdm=False):
    """Train for one epoch."""
    epoch_batch_losses = []
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Loss: N/A")
    
    for batch_idx, (inputs, targets, texts) in enumerate(pbar):
        # Get hidden states from vLLM
        outputs = llm.encode(texts, use_tqdm=use_tqdm)
        
        # Process hidden states
        eos_hidden_states = extract_eos_hidden_states(outputs, texts, inputs, tokenizer, device)
        
        # Forward pass
        pred_conf = model(eos_hidden_states)
        loss = criterion(pred_conf, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_batch_losses.append(loss.item())
        pbar.set_description(f"Training Loss: {loss.item():.4f}")
        
        # Memory management
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    return epoch_loss / len(train_loader), epoch_batch_losses


def train_model(model, llm, train_loader, optimizer, criterion, tokenizer, device, num_epochs=5, use_tqdm=False):
    """Full training loop."""
    print(f"Training for {num_epochs} epochs over {len(train_loader.dataset)} samples "
          f"({len(train_loader)} batches per epoch).")
    
    epoch_losses = []
    all_batch_losses = []
    for epoch in range(num_epochs):
        avg_loss, batch_losses = train_epoch(
            model=model,
            llm=llm,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device,
            use_tqdm=use_tqdm
        )
        
        epoch_losses.append(avg_loss)
        all_batch_losses.extend(batch_losses)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    print("Training complete.")
    return epoch_losses, all_batch_losses


def main(dataset_name, llm_id, wconf_dir_path, nshots, max_seq_length, epochs, gpu_memory, tensor_parallel, vllm_task, batch_q, nruns, seed):
    dataset = load_wconf_dataset(wconf_dir_path, llm_id)
    tokenizer = initialize_tokenizer(llm_id, max_seq_length)
    few_shot_prompt = load_few_shot_examples(wconf_dir_path, nshots)
    system_prompt = load_system_prompt(dataset_name)
    dataset_flat = flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer)
    assert_flatten_dataset(dataset_flat, tokenizer)

    t2l = Talk2LLM(model_id=llm_id, task=vllm_task, gpu_memory_utilization=gpu_memory, tensor_parallel_size=tensor_parallel, the_seed=seed)
    hidden_size = get_last_hidden_state_size(llm_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    confidence_head, optimizer, criterion = setup_model_and_optimizer(hidden_size, device)

    flattened_batch_size = batch_q * nruns
    collate_fn = create_collate_fn(tokenizer, device, max_seq_length)
    train_loader, test_loader = setup_data_loaders(dataset_flat, flattened_batch_size, collate_fn)
    
    epoch_losses, all_batch_losses = train_model(
        model=confidence_head,
        llm=t2l.llm,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        num_epochs=epochs,
        use_tqdm=False
    )

    save_path = save_training_artifacts(
        model_name=llm_id.split('/')[-1],
        confidence_head=confidence_head,
        optimizer=optimizer,
        tokenizer=tokenizer,
        epoch_losses=epoch_losses,
        batch_losses=all_batch_losses
    )
    print('=' * 50)
    print(f"✅ Done! Training artifacts saved to: {save_path}")
    print('=' * 50)


if __name__ == "__main__":
    main(dataset_name=args.dataset_name, 
         llm_id=args.llm_id, 
         wconf_dir_path=args.wconf_dir_path, 
         nshots=args.nshots, 
         max_seq_length=args.max_seq_length, 
         epochs=args.epochs, 
         gpu_memory=args.gpu_memory, 
         tensor_parallel=args.tensor_parallel, 
         vllm_task=args.vllm_task, 
         batch_q=args.batch_q, 
         nruns=args.nruns, 
         seed=args.seed)

# python last-hidden-state-train.py --dataset_name "GSM8K" --llm_id "meta-llama/Llama-3.2-1B-Instruct" --wconf_dir_path "../data/GSM8K/" --nshots 10 --visible_cudas "5" --gpu_memory 0.8 --tensor_parallel 1 --max_seq_length 2300 --vllm_task "reward" --epochs 5 --batch_q 30 --nruns 30 --seed 23