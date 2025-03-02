import os
import re
import jsonlines
import json
from tqdm import tqdm
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_visible_cudas(gpu_ids):
    print(f"Visible CUDAs before setting: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"Visible CUDAs after setting: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def gsm8k_system_prompt():
    return "You are a mathematics expert. You will be given a mathematics problem which you need to solve. Provide the final answer clearly at the end in the format: #### <final answer>."

def load_system_prompt(dataset_name):
    if dataset_name == "GSM8K":
        return gsm8k_system_prompt()
    return None

def extract_gsm8k_final_answer(answer_text):
    """
    Extracts the final numerical answer from a GSM8K-style response.
    - Finds the last occurrence of '####' and extracts the first valid number after it.
    - Handles cases where extra text follows the number.
    - Returns None if no valid number is found.
    """
    parts = answer_text.strip().split("####")
    
    if len(parts) > 1:
        # Get the last part after the last '####'
        last_segment = parts[-1].strip()

        # Use regex to extract the first valid number
        match = re.search(r"[-+]?\d*\.?\d+", last_segment)
        if match:
            return match.group()  # Return the extracted number as a string
    
    return None  # Return None if format is unexpected

# Load few-shot examples for prompting
def load_few_shot_examples(data_path, nshots):
    file_path = os.path.join(data_path, f"{nshots}-shots.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_jsonl(file_path):
    """Loads a JSONL file and returns a list of dictionaries."""
    try:
        with jsonlines.open(file_path, "r") as reader:
            return [obj for obj in reader]  # Read all JSON objects into a list
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_dataset_splits(base_dir, postfix, splits=("train", "val", "test")):
    """
    Loads dataset splits from JSONL files.
    
    Args:
        base_dir (str): The base directory where dataset files are stored.
        splits (tuple): Dataset splits to load (default: train, val, test).

    Returns:
        dict: A dictionary with split names as keys and loaded data as values.
    """
    dataset = {}
    for split in splits:
        file_path = os.path.join(base_dir, f"{split}{postfix}.jsonl")
        dataset[split] = load_jsonl(file_path)    
    return dataset

def load_wconf_dataset(wconf_dir_path, postfix):
    """
    Load the wconf data for the given LLM model.
    
    Args:
        wconf_dir_path (str): Path to the directory containing the wconf files.
        postfix (str): Postfix used in input file names, e.g. 'wconf' for train_wconf.jsonl, etc.
    
    Returns:
        dict: A dictionary with split names as keys and loaded wconf data as values.
    """
    # Load dataset splits
    dataset = load_dataset_splits(wconf_dir_path, postfix)
    # Print dataset sizes
    for split, data in dataset.items():
        print(f"{split.capitalize()} samples: {len(data)}")
    return dataset

def create_flattened_dataset_conf(samples, few_shot_prompt, system_prompt, tokenizer):
    """
    For each sample in 'samples' (each representing one question), create
    a new record for each run (i.e. each LLM output in confidence_runs) using a chat conversation.
    
    The conversation is constructed as follows:
      - System: system_prompt
      - User: few_shot_prompt + "\nQuestion: {question}\nAnswer:"
      - Assistant: the LLM's generated answer (taken from run["full_llm_answer"])
    
    Each new record contains:
      - "input_str": the formatted conversation (using the chat template)
      - "empirical_confidence": the aggregated target (same for all runs of that question)
    
    Args:
        samples (list): List of original samples from the dataset.
        few_shot_prompt (str): The few-shot examples.
        system_prompt (str): The system instruction.
        tokenizer: The tokenizer that supports chat formatting.
        num_runs (int or None): If set, use only the first num_runs from each sample's confidence_runs.
    
    Returns:
        flattened_data (list of dict): The new flattened dataset.
    """
    flattened_data = []
    for sample in tqdm(samples, desc='Creating Flattened Dataset'):
        # Build the user prompt (do not include any ground truth answer here)
        if few_shot_prompt.endswith("\n\n"):
            user_prompt = f"{few_shot_prompt}Question: {sample['question']}\nAnswer: "
        else:
            user_prompt = f"{few_shot_prompt}\n\nQuestion: {sample['question']}\nAnswer: "

        # If num_runs is specified, restrict to that many runs; otherwise, use all runs.
        runs = sample["confidence_runs"]
        for run in runs:
            assistant_message = run["full_llm_answer"].strip()
            assistant_message = assistant_message.split('Assistant Message: ')[-1]
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_message}
            ]
            formatted_str = tokenizer.apply_chat_template(conversation, tokenize=False)
            new_record = {
                "record_hash_id": sample["record_hash_id"],
                "input_str": formatted_str,
                "empirical_confidence": sample["empirical_confidence"],
                "record_hash_ids": sample["record_hash_ids"],
                "run_hash_id": run["run_hash_id"],
            }
            flattened_data.append(new_record)
    return flattened_data


def is_predictionn_correct(pred, target):
    if str(pred) == str(target):
        return 1
    return 0


def create_flattened_dataset_true(samples, few_shot_prompt, system_prompt, tokenizer):
    """
    For each sample in 'samples' (each representing one question), create
    a new record for each run (i.e. each LLM output in confidence_runs) using a chat conversation.
    
    The conversation is constructed as follows:
      - System: system_prompt
      - User: few_shot_prompt + "\nQuestion: {question}\nAnswer:"
      - Assistant: the LLM's generated answer (taken from run["full_llm_answer"])
    
    Each new record contains:
      - "input_str": the formatted conversation (using the chat template)
      - "empirical_confidence": the aggregated target (same for all runs of that question)
    
    Args:
        samples (list): List of original samples from the dataset.
        few_shot_prompt (str): The few-shot examples.
        system_prompt (str): The system instruction.
        tokenizer: The tokenizer that supports chat formatting.
        num_runs (int or None): If set, use only the first num_runs from each sample's confidence_runs.
    
    Returns:
        flattened_data (list of dict): The new flattened dataset.
    """
    flattened_data = []
    for sample in tqdm(samples, desc='Creating Flattened Dataset'):
        # Build the user prompt (do not include any ground truth answer here)
        if few_shot_prompt.endswith("\n\n"):
            user_prompt = f"{few_shot_prompt}Question: {sample['question']}\nAnswer: "
        else:
            user_prompt = f"{few_shot_prompt}\n\nQuestion: {sample['question']}\nAnswer: "
        # If num_runs is specified, restrict to that many runs; otherwise, use all runs.
        runs = sample["confidence_runs"]
        for run in runs:
            correct_or_not = is_predictionn_correct(run["final_llm_answer"], sample["final_answer"])
            assistant_message = run["full_llm_answer"].strip()
            assistant_message = assistant_message.split('Assistant Message: ')[-1]
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_message}
            ]
            formatted_str = tokenizer.apply_chat_template(conversation, tokenize=False)
            new_record = {
                "record_hash_id": sample["record_hash_id"],
                "input_str": formatted_str,
                "correct_or_not": correct_or_not,
                "record_hash_ids": sample["record_hash_ids"],
                "run_hash_id": run["run_hash_id"],
            }
            flattened_data.append(new_record)
    return flattened_data


def flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer, goal='conf'):
    """
    Flatten the dataset for train, val, and test splits.

    Args:
        dataset (dict): The dataset with train, val, and test splits.
        few_shot_prompt (str): The few-shot examples.
        system_prompt (str): The system instruction.
        tokenizer: The tokenizer that supports chat formatting.

    Returns:
        dataset_flat (dict): The flattened dataset with train, val, and test splits.
    """
    # Flatten the dataset for train val test and define dataset_flat
    dataset_flat = {}
    for split, data in dataset.items():
        if goal == 'conf':
            dataset_flat[split] = create_flattened_dataset_conf(data, few_shot_prompt, system_prompt, tokenizer)
        elif goal == 'true':
            dataset_flat[split] = create_flattened_dataset_true(data, few_shot_prompt, system_prompt, tokenizer)
    # Print dataset sizes
    for split, data in dataset_flat.items():
        print(f"{split.capitalize()} flattened samples: {len(data)}")
    return dataset_flat

def assert_flatten_dataset(dataset_flat, tokenizer):
    for split, data in dataset_flat.items():
        for sample in data:
            assert "<|start_header_id|>system<|end_header_id|>" in sample['input_str']
            assert "<|start_header_id|>user<|end_header_id|>" in sample['input_str']
            assert "<|start_header_id|>assistant<|end_header_id|>" in sample['input_str']
            eos_token = tokenizer.eos_token
            assert sample['input_str'].endswith(eos_token)
    print('All header ids found in all samples of all splits of dataset flat')
    print('All samples of all splits of dataset flat have eos token at the end')


def get_last_hidden_state_size(llm_id):
    """
    Get the size of the last hidden state of the LLM model.
    
    Args:
        llm_id (str): The ID of the LLM model.
    
    Returns:
        int: The size of the last hidden state.
    """
    if '1B' in llm_id:
        return 2048
    elif '3B' in llm_id:
        return 3072
    elif '8B' in llm_id:
        return 4096
    return None


def create_collate_fn(tokenizer, device, max_seq_length, goal='conf', return_ids=False):
    def collate_fn(batch):
        texts = [item["input_str"] for item in batch]
        if goal == 'conf':
            targets = [item["empirical_confidence"] for item in batch]
        elif goal == 'true':
            targets = [item["correct_or_not"] for item in batch]
        run_ids = [item["run_hash_id"] for item in batch]
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_seq_length
        )
        
        # Move tensors to the device
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        
        # Generate truncated/padded text from tokenized inputs.
        # We decode each sequence; set skip_special_tokens=False if you want to see special tokens.
        texts_trunc_pad = [
            tokenizer.decode(input_ids, skip_special_tokens=False)
            for input_ids in inputs["input_ids"]
        ]

        if not return_ids:    
            return inputs, targets, texts, texts_trunc_pad
        else:
            return inputs, targets, texts, texts_trunc_pad, run_ids
    return collate_fn



def setup_data_loaders(dataset_flat, batch_size, collate_fn):
    hf_train_dataset = Dataset.from_list(dataset_flat["train"])
    hf_val_dataset = Dataset.from_list(dataset_flat["val"])
    hf_test_dataset = Dataset.from_list(dataset_flat["test"])
    train_loader = DataLoader(
        hf_train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        hf_val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    test_loader = DataLoader(
        hf_test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )    
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def initialize_tokenizer(llm_id, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.model_max_length = max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_losses(save_path, epoch_losses, batch_losses):
    """Save training losses to JSON file."""
    losses_dict = {
        'epoch_losses': epoch_losses,
        'batch_losses': batch_losses
    }
    losses_path = os.path.join(save_path, 'training_losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Training losses saved to {losses_path}")


# -------------------------------
# Function to plot and save metric evolution.
def plot_all_metrics(metrics_history, output_dir, num_epochs, goal='conf'):
    try:
        print("Plotting confidence metrics...")
        epochs_arr = np.arange(1, num_epochs+1)
        
        # Plot training, validation, and test loss.
        plt.figure(figsize=(8,6))
        plt.plot(epochs_arr, metrics_history["train_loss"], marker='o', label="Train Loss")
        for x, y in zip(epochs_arr, metrics_history["train_loss"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.plot(epochs_arr, metrics_history["val_loss"], marker='o', label="Val Loss")
        for x, y in zip(epochs_arr, metrics_history["val_loss"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.plot(epochs_arr, metrics_history["test_loss"], marker='o', label="Test Loss")
        for x, y in zip(epochs_arr, metrics_history["test_loss"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend(loc="upper center")
        plt.savefig(os.path.join(output_dir, "loss_over_epochs.png"))
        plt.close()
        
        # Plot BCE (validation and test).
        plt.figure(figsize=(8,6))
        plt.plot(epochs_arr, metrics_history["BCE"]["val"], marker='o', label="Val BCE Loss")
        for x, y in zip(epochs_arr, metrics_history["BCE"]["val"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.plot(epochs_arr, metrics_history["BCE"]["test"], marker='o', label="Test BCE Loss")
        for x, y in zip(epochs_arr, metrics_history["BCE"]["test"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("BCE Loss Over Epochs")
        plt.legend(loc="upper center")
        plt.savefig(os.path.join(output_dir, "bce_over_epochs.png"))
        plt.close()
        
        # Plot Brier Score.
        plt.figure(figsize=(8,6))
        plt.plot(epochs_arr, metrics_history["Brier"]["val"], marker='o', label="Val Brier Score")
        for x, y in zip(epochs_arr, metrics_history["Brier"]["val"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.plot(epochs_arr, metrics_history["Brier"]["test"], marker='o', label="Test Brier Score")
        for x, y in zip(epochs_arr, metrics_history["Brier"]["test"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.xlabel("Epoch")
        plt.ylabel("Brier Score")
        plt.title("Brier Score Over Epochs")
        plt.legend(loc="upper center")
        plt.savefig(os.path.join(output_dir, "brier_over_epochs.png"))
        plt.close()
        
        # Plot ECE (both versions).
        plt.figure(figsize=(8,6))
        plt.plot(epochs_arr, metrics_history["ECE"]["val"], marker='o', label="Val ECE")
        for x, y in zip(epochs_arr, metrics_history["ECE"]["val"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.plot(epochs_arr, metrics_history["ECE"]["test"], marker='o', label="Test ECE")
        for x, y in zip(epochs_arr, metrics_history["ECE"]["test"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.xlabel("Epoch")
        plt.ylabel("ECE")
        plt.title("ECE Over Epochs")
        plt.legend(loc="upper center")
        plt.savefig(os.path.join(output_dir, "ece_over_epochs.png"))
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.plot(epochs_arr, metrics_history["ECE Manual"]["val"], marker='o', label="Val ECE Manual")
        for x, y in zip(epochs_arr, metrics_history["ECE Manual"]["val"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.plot(epochs_arr, metrics_history["ECE Manual"]["test"], marker='o', label="Test ECE Manual")
        for x, y in zip(epochs_arr, metrics_history["ECE Manual"]["test"]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
        plt.xlabel("Epoch")
        plt.ylabel("ECE Manual")
        plt.title("Manual ECE Over Epochs")
        plt.legend(loc="upper center")
        plt.savefig(os.path.join(output_dir, "ece_manual_over_epochs.png"))
        plt.close()

        # If goal is "true", plot additional classification metrics.
        if goal == "true":
            print("Plotting classification metrics...")
            # Plot Accuracy.
            plt.figure(figsize=(8,6))
            plt.plot(epochs_arr, metrics_history["accuracy"]["val"], marker='o', label="Val Accuracy")
            for x, y in zip(epochs_arr, metrics_history["accuracy"]["val"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.plot(epochs_arr, metrics_history["accuracy"]["test"], marker='o', label="Test Accuracy")
            for x, y in zip(epochs_arr, metrics_history["accuracy"]["test"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Over Epochs")
            plt.legend(loc="upper center")
            plt.savefig(os.path.join(output_dir, "accuracy_over_epochs.png"))
            plt.close()
            print('Accuracy saved at:', os.path.join(output_dir, "accuracy_over_epochs.png"))

            # Plot Precision.
            plt.figure(figsize=(8,6))
            plt.plot(epochs_arr, metrics_history["precision"]["val"], marker='o', label="Val Precision")
            for x, y in zip(epochs_arr, metrics_history["precision"]["val"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.plot(epochs_arr, metrics_history["precision"]["test"], marker='o', label="Test Precision")
            for x, y in zip(epochs_arr, metrics_history["precision"]["test"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.xlabel("Epoch")
            plt.ylabel("Precision")
            plt.title("Precision Over Epochs")
            plt.legend(loc="upper center")
            plt.savefig(os.path.join(output_dir, "precision_over_epochs.png"))
            plt.close()
            
            # Plot Recall.
            plt.figure(figsize=(8,6))
            plt.plot(epochs_arr, metrics_history["recall"]["val"], marker='o', label="Val Recall")
            for x, y in zip(epochs_arr, metrics_history["recall"]["val"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.plot(epochs_arr, metrics_history["recall"]["test"], marker='o', label="Test Recall")
            for x, y in zip(epochs_arr, metrics_history["recall"]["test"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.xlabel("Epoch")
            plt.ylabel("Recall")
            plt.title("Recall Over Epochs")
            plt.legend(loc="upper center")
            plt.savefig(os.path.join(output_dir, "recall_over_epochs.png"))
            plt.close()
            
            # Plot F1 Score.
            plt.figure(figsize=(8,6))
            plt.plot(epochs_arr, metrics_history["f1"]["val"], marker='o', label="Val F1")
            for x, y in zip(epochs_arr, metrics_history["f1"]["val"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.plot(epochs_arr, metrics_history["f1"]["test"], marker='o', label="Test F1")
            for x, y in zip(epochs_arr, metrics_history["f1"]["test"]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title("F1 Score Over Epochs")
            plt.legend(loc="upper center")
            plt.savefig(os.path.join(output_dir, "f1_over_epochs.png"))
            plt.close()

            try:            
                # Plot ROC AUC.
                plt.figure(figsize=(8,6))
                plt.plot(epochs_arr, metrics_history["aucroc"]["val"], marker='o', label="Val AUC-ROC")
                for x, y in zip(epochs_arr, metrics_history["aucroc"]["val"]):
                    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
                plt.plot(epochs_arr, metrics_history["aucroc"]["test"], marker='o', label="Test AUC-ROC")
                for x, y in zip(epochs_arr, metrics_history["aucroc"]["test"]):
                    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
                plt.xlabel("Epoch")
                plt.ylabel("AUC-ROC")
                plt.title("AUC-ROC Over Epochs")
                plt.legend(loc="upper center")
                plt.savefig(os.path.join(output_dir, "aucroc_over_epochs.png"))
                plt.close()
            except Exception as e:
                pass

            try:
                # Plot AUC PR.
                plt.figure(figsize=(8,6))
                plt.plot(epochs_arr, metrics_history["aucpr"]["val"], marker='o', label="Val AUC-PR")
                for x, y in zip(epochs_arr, metrics_history["aucpr"]["val"]):
                    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
                plt.plot(epochs_arr, metrics_history["aucpr"]["test"], marker='o', label="Test AUC-PR")
                for x, y in zip(epochs_arr, metrics_history["aucpr"]["test"]):
                    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
                plt.xlabel("Epoch")
                plt.ylabel("AUC-PR")
                plt.title("AUC-PR Over Epochs")
                plt.legend(loc="upper center")
                plt.savefig(os.path.join(output_dir, "aucpr_over_epochs.png"))
                plt.close()
            except Exception as e:
                pass            
        print("All metrics plots saved successfully.")
    except Exception as e:
        print(f"Error in plot_all_metrics: {e}")

# -------------------------------
# Define evaluation functions.
def compute_brier_score(y_true, y_pred):
    """Compute Brier Score (mean squared error for probabilities)."""
    try:
        return np.mean((y_pred - y_true) ** 2)
    except Exception as e:
        print(f"Error in compute_brier_score: {e}")
        return None

def compute_ece(y_true, y_pred, bins=10):
    """Compute Expected Calibration Error (ECE) using netcal."""
    try:
        ece_metric = ECE(bins=bins)
        return ece_metric.measure(y_pred, y_true)
    except Exception as e:
        print(f"Error in compute_ece: {e}")
        return None

def compute_ece_manually(y_true, y_pred, bins=10):
    """Compute Expected Calibration Error (ECE) manually."""
    try:
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges, right=True) - 1
        ece = 0.0
        for i in range(bins):
            mask = bin_indices == i
            if np.sum(mask) == 0:
                continue
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            bin_conf = np.mean(bin_pred)
            bin_acc = np.mean(bin_true)
            ece += np.abs(bin_conf - bin_acc) * np.sum(mask)
        ece /= len(y_true)
        return ece
    except Exception as e:
        print(f"Error in compute_ece_manually: {e}")
        return None
    
def compute_classification_metrics(y_true, y_pred):
    # Convert probability predictions to binary predictions using a threshold of 0.5
    y_pred_binary = [1 if prob >= 0.5 else 0 for prob in y_pred]
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # For AUC metrics, use the original probability predictions
    aucroc, aucpr = None, None
    try:
        aucroc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        pass
    try:
        aucpr = average_precision_score(y_true, y_pred)
    except Exception as e:
        pass
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'aucroc': aucroc,
        'aucpr': aucpr
    }


def compute_metrics(y_true, y_pred, goal='conf'):
    """Compute metrics: BCE Loss, Brier Score, and both versions of ECE."""
    eps = 1e-7
    bce_loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    brier = compute_brier_score(y_true, y_pred)
    ece = compute_ece(y_true, y_pred)
    ece_manual = compute_ece_manually(y_true, y_pred)
    if goal == 'conf':         
        return {"BCE Loss": bce_loss, "Brier Score": brier, "ECE": ece, "ECE Manual": ece_manual}
    elif goal == 'true':
        classification_metrics = compute_classification_metrics(y_true, y_pred)
        return {**classification_metrics, "BCE Loss": bce_loss, "Brier Score": brier, "ECE": ece, "ECE Manual": ece_manual}

def save_predictions_and_model(preds_dir, epoch_dir, epoch, val_targets, val_preds, test_targets, test_preds, conf_head):
    """Saves validation and test targets/predictions as .npy files and model weights as .pt file."""
    
    # Save validation and test targets/predictions
    np.save(os.path.join(preds_dir, f"val_targets_epoch{epoch+1}.npy"), val_targets)
    np.save(os.path.join(preds_dir, f"val_preds_epoch{epoch+1}.npy"), val_preds)
    np.save(os.path.join(preds_dir, f"test_targets_epoch{epoch+1}.npy"), test_targets)
    np.save(os.path.join(preds_dir, f"test_preds_epoch{epoch+1}.npy"), test_preds)
    
    # Save model weights for this epoch
    torch.save(conf_head.state_dict(), os.path.join(epoch_dir, f"conf_head_epoch{epoch+1}.pt"))