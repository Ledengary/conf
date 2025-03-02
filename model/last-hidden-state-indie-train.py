import argparse
parser = argparse.ArgumentParser(description="Compute empirical confidence for GSM8K using vLLM")
parser.add_argument("--visible_cudas", type=str, help="CUDA devices to use")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--llm_id", type=str, required=True, help="ID of the LLM model")
parser.add_argument("--gpu_memory", type=float, default=0.5, help="GPU memory utilization")
parser.add_argument("--tensor_parallel", type=int, default=2, help="Tensor parallel size")
parser.add_argument("--wconf_dir_path", type=str, required=True, help="Path to the directory containing the wconf files")
parser.add_argument("--shots_dir_path", type=str, required=True, help="Path to the directory containing the few-shot examples")
parser.add_argument("--postfix", type=str, required=True, help="Postfix used in input file names, e.g. '_wconf', '_wconf_wid', etc.")
parser.add_argument("--nshots", type=int, default=10, help="Number of shots in few-shot prompting")
parser.add_argument("--batch_q", type=int, default=30, help="number of unique questions per batch")
parser.add_argument("--nruns", type=int, default=30, help="Number of LLM runs per sample")
parser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length for the tokenizer")
parser.add_argument("--vllm_task", type=str, default="reward", help="Task for vLLM")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility")
parser.add_argument("--output_dir", type=str, required=True, help="Directory where training artifacts will be stored")
parser.add_argument("--final_hidden_states_path", type=str, required=True, help="Directory where final hidden state embeddings are stored (with train, val, test subdirs)")
parser.add_argument("--goal", type=str, required=True, help="Prediction goal ('conf' or 'true')")
# New hyperparameters for ConfidenceHeadStackedInput
parser.add_argument("--conv_channels", type=int, default=32, help="Number of channels for the convolution in the confidence head")
parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for the convolution in the confidence head")
parser.add_argument("--conv_activation", type=str, default="relu", help="Activation function for the convolution ('relu', 'gelu', etc.)")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the confidence head")
args = parser.parse_args()

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas
import argparse
import os
import sys
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
from collections import defaultdict
import json

# ---------------------------
# Add utils directory and import utilities
# ---------------------------
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (load_wconf_dataset, load_few_shot_examples, load_system_prompt,
                     flatten_dataset, assert_flatten_dataset, get_last_hidden_state_size,
                     initialize_tokenizer, plot_all_metrics, compute_metrics, seed_everything)
seed_everything(args.seed)
print(f"Random seed set to {args.seed}")
sys.path = original_sys_path

# ---------------------------
# Import the probe classifier model.
# ---------------------------
from confhead import ConfidenceHeadFlattenInput, ConfidenceHeadFlattenInputAug

# ---------------------------
# Custom Dataset for Precomputed Embeddings (with preloading on CPU)
# ---------------------------
def print_dataset_statistics(records, targets, goal):
    """
    Prints dataset statistics:
      - Total number of samples.
      - Mean number of runs per question and number of unique questions (if 'record_hash_id' exists).
      - For goal 'true': distribution of correct answers and histogram of per-question correct percentages.
      - For goal 'conf': histogram of empirical confidence values.
    """
    print(f"Total samples: {len(records)}")
    
    # If each record contains a 'record_hash_id', compute per-question statistics.
    if all("record_hash_id" in rec for rec in records):
        question_runs = defaultdict(list)
        for rec in records:
            qid = rec["record_hash_id"]
            question_runs[qid].append(rec["run_hash_id"])
        mean_runs = np.mean([len(runs) for runs in question_runs.values()])
        print(f"Mean number of runs per question: {mean_runs:.2f}")
        print(f"Total number of unique questions: {len(question_runs)}")
        
        if goal == "true":
            # Compute per-question percentage of correct answers.
            question_correct = defaultdict(list)
            for rec in records:
                qid = rec["record_hash_id"]
                question_correct[qid].append(float(rec["correct_or_not"]))
            per_question_percents = [np.mean(vals) for vals in question_correct.values()]
            hist, bin_edges = np.histogram(per_question_percents, bins=10)
            print("Histogram of per-question percentage of correct answers:")
            print("Bins:", bin_edges)
            print("Counts:", hist)
    else:
        print("No 'record_hash_id' found in records; skipping per-question statistics.")
    
    # Print distribution of targets.
    targets_arr = np.array(targets)
    if goal == "true":
        print("Distribution of 'correct_or_not':")
        unique, counts = np.unique(targets_arr, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Value {u}: count {c}, percentage {100 * c / len(targets_arr):.2f}%")
    else:
        hist, bin_edges = np.histogram(targets_arr, bins=10)
        print("Distribution of empirical confidences:")
        print("Bins:", bin_edges)
        print("Counts:", hist)

class EmbeddingDataset(Dataset):
    """
    Loads all precomputed final hidden state embeddings into memory on the CPU.
    Returns (embedding, target) for each record. Each record must include:
      - "run_hash_id": to locate the embedding file.
      - For goal 'conf': "empirical_confidence" as target.
      - For goal 'true': "correct_or_not" as target.
    Optionally, if available, each record should include "record_hash_id" for grouping by question.
    """
    def __init__(self, records, embeddings_dir, goal):
        self.records = records
        self.embeddings_dir = embeddings_dir  # e.g. final_hidden_states_path/split/model_name
        self.goal = goal
        self.embeddings = []
        self.targets = []
        
        for rec in self.records:
            run_id = rec["run_hash_id"]
            emb_path = os.path.join(self.embeddings_dir, f"{run_id}.pt")
            embedding = torch.load(emb_path, map_location="cpu", weights_only=True)
            self.embeddings.append(embedding)
            if self.goal == 'conf':
                self.targets.append(float(rec["empirical_confidence"]))
            else:
                self.targets.append(float(rec["correct_or_not"]))
        
        print(f"Loaded {len(self.embeddings)} embeddings.")
        # Call the separate function to print dataset statistics.
        print_dataset_statistics(self.records, self.targets, self.goal)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


# ---------------------------
# Utility function to build DataLoaders for a given split.
# ---------------------------
def get_dataloader(flattened_records, embeddings_base_dir, split, batch_size, goal):
    split_dir = os.path.join(embeddings_base_dir, split)
    dataset = EmbeddingDataset(flattened_records, split_dir, goal)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))
    return loader

# ---------------------------
# Main training and evaluation functions.
# ---------------------------
def evaluate_model(loader, model, device, goal="conf"):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.BCELoss()
    count = 0
    with torch.no_grad():
        for embeddings, targets in tqdm(loader, desc="Evaluating", leave=False):
            embeddings = embeddings.to(device).float()
            targets = torch.as_tensor(targets, device=device, dtype=torch.float)
            preds = model(embeddings)
            loss = criterion(preds, targets)
            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / count
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), goal=goal)
    return avg_loss, metrics, np.array(all_targets), np.array(all_preds)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()
    for embeddings, targets in tqdm(loader, desc="Training", leave=False):
        embeddings = embeddings.to(device).float()
        targets = torch.as_tensor(targets, device=device, dtype=torch.float)
        optimizer.zero_grad()
        preds = model(embeddings)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
    return total_loss / len(loader.dataset)

# ---------------------------
# Main function.
# ---------------------------
def main(dataset_name, goal, llm_id, wconf_dir_path, postfix, shots_dir_path, nshots, 
         max_seq_length, epochs, batch_q, nruns, seed, output_dir, conv_channels, kernel_size, conv_activation, dropout, final_hidden_states_path):
    dataset = load_wconf_dataset(wconf_dir_path, postfix)
    tokenizer = initialize_tokenizer(llm_id, max_seq_length)
    few_shot_prompt = load_few_shot_examples(shots_dir_path, nshots)
    system_prompt = load_system_prompt(dataset_name)
    dataset_flat = flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer, goal=goal)
    assert_flatten_dataset(dataset_flat, tokenizer)
    
    train_records = dataset_flat["train"]
    val_records = dataset_flat["val"]
    test_records = dataset_flat["test"]
    
    batch_size = batch_q * nruns
    train_loader = get_dataloader(train_records, final_hidden_states_path, "train", batch_size, goal)
    val_loader = get_dataloader(val_records, final_hidden_states_path, "val", batch_size, goal)
    test_loader = get_dataloader(test_records, final_hidden_states_path, "test", batch_size, goal)
    
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = get_last_hidden_state_size(llm_id)
    print(f"Using device: {device} with hidden size: {hidden_size}")
    
    probe = ConfidenceHeadFlattenInput(hidden_size)
    probe.to(device)
    optimizer = Adam(probe.parameters(), lr=1e-4)
    
    # Setup directories for saving artifacts.
    model_name = llm_id.replace("/", "-")
    output_dir = os.path.join(args.output_dir, model_name, args.goal, 
                              f"conv{args.conv_channels}_k{args.kernel_size}_act{args.conv_activation}_drop{args.dropout}_dt{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    epoch_dir = os.path.join(output_dir, "epoch_weights")
    os.makedirs(epoch_dir, exist_ok=True)
    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    preds_dir = os.path.join(output_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write(str(vars(args)))
    
    # Initialize metrics_history with loss keys and empty dicts for additional metrics.
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "BCE": {"val": [], "test": []},
        "Brier": {"val": [], "test": []},
        "ECE": {"val": [], "test": []},
        "ECE Manual": {"val": [], "test": []},
    }
    if args.goal == "true":
        metrics_history.update({
            "accuracy": {"val": [], "test": []},
            "precision": {"val": [], "test": []},
            "recall": {"val": [], "test": []},
            "f1": {"val": [], "test": []},
            "aucroc": {"val": [], "test": []},
            "aucpr": {"val": [], "test": []},
        })

    best_model_path = os.path.join(best_dir, "probe_best.pt")
    best_val_loss = None
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss = train_epoch(probe, train_loader, optimizer, device)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
        # Evaluate on validation set.
        val_loss, val_metrics, _, _ = evaluate_model(val_loader, probe, device, args.goal)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate on test set.
        test_loss, test_metrics, _, _ = evaluate_model(test_loader, probe, device, args.goal)
        print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}")
        
        # Log losses.
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["test_loss"].append(test_loss)
        
        # Log additional metrics (BCE loss is taken as avg_loss here).
        metrics_history["BCE"]["val"].append(val_loss)
        metrics_history["BCE"]["test"].append(test_loss)
        metrics_history["Brier"]["val"].append(val_metrics["Brier Score"])
        metrics_history["Brier"]["test"].append(test_metrics["Brier Score"])
        metrics_history["ECE"]["val"].append(val_metrics["ECE"])
        metrics_history["ECE"]["test"].append(test_metrics["ECE"])
        metrics_history["ECE Manual"]["val"].append(val_metrics["ECE Manual"])
        metrics_history["ECE Manual"]["test"].append(test_metrics["ECE Manual"])
        
        # For goal "true", log additional classification metrics.
        if args.goal == "true":
            metrics_history["accuracy"]["val"].append(val_metrics["accuracy"])
            metrics_history["accuracy"]["test"].append(test_metrics["accuracy"])
            metrics_history["precision"]["val"].append(val_metrics["precision"])
            metrics_history["precision"]["test"].append(test_metrics["precision"])
            metrics_history["recall"]["val"].append(val_metrics["recall"])
            metrics_history["recall"]["test"].append(test_metrics["recall"])
            metrics_history["f1"]["val"].append(val_metrics["f1"])
            metrics_history["f1"]["test"].append(test_metrics["f1"])
            metrics_history["aucroc"]["val"].append(val_metrics["aucroc"])
            metrics_history["aucroc"]["test"].append(test_metrics["aucroc"])
            metrics_history["aucpr"]["val"].append(val_metrics["aucpr"])
            metrics_history["aucpr"]["test"].append(test_metrics["aucpr"])
        
        # Update best model based on validation loss.
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(probe.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch+1}")

    
    print("Loading best model for test evaluation...")
    probe.load_state_dict(torch.load(best_model_path))
    final_test_loss, final_test_metrics, test_targets, test_preds = evaluate_model(test_loader, probe, device, goal)
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print("Final Test Metrics:")
    for key, value in final_test_metrics.items():
        print(f"  {key}: {value:.4f}")

    np.save(os.path.join(preds_dir, "test_targets.npy"), test_targets)
    np.save(os.path.join(preds_dir, "test_preds.npy"), test_preds)
    
    plot_all_metrics(metrics_history, output_dir, epochs, goal=goal)
    
    # Save final metrics
    results_dir = os.path.join(best_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save final test metrics
    with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serialized_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in final_test_metrics.items()}
        json.dump(serialized_metrics, f, indent=2)

    print(f"Final metrics saved in {os.path.join(results_dir, 'final_metrics.json')}")

    
    print(f"Training complete. Artifacts saved in {output_dir}")

if __name__ == "__main__":
    main(
        dataset_name=args.dataset_name,
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
        output_dir=args.output_dir,
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        conv_activation=args.conv_activation,
        dropout=args.dropout,
        final_hidden_states_path=args.final_hidden_states_path
    )

# python last-hidden-state-indie-train.py \
#   --visible_cudas "0,1,2" \
#   --dataset_name "GSM8K" \
#   --goal "true" \
#   --llm_id "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" \
#   --wconf_dir_path "../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/subset" \
#   --postfix "_wconf_wid_sub" \
#   --shots_dir_path "../data/GSM8K/" \
#   --nshots 10 \
#   --max_seq_length 1024 \
#   --epochs 5 \
#   --batch_q 1 \
#   --nruns 30 \
#   --seed 23 \
#   --output_dir "../storage/trained_models" \
#   --final_hidden_states_path "../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/final_hidden_states" \
#   --conv_channels 32 \
#   --kernel_size 1 \
#   --conv_activation "relu" \
#   --dropout 0.1