#!/usr/bin/env python
import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

# Parse command-line arguments.
parser = argparse.ArgumentParser(
    description="Extract attention representations from an LLM using Accelerate."
)
parser.add_argument("--visible_cudas", type=str, required=True,
                    help="CUDA devices to use (e.g., '0,1,2')")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="Name of the dataset (e.g., 'GSM8K')")
parser.add_argument("--llm_id", type=str, required=True,
                    help="ID or path of the LLM model (e.g., 'meta-llama/Llama-3.1-8B-Instruct')")
parser.add_argument("--wconf_dir_path", type=str, required=True,
                    help="Directory path containing the wconf JSONL files (organized as {wconf_dir_path}/{llm_id}/{split}_wconf.jsonl)")
parser.add_argument("--wconf_postfix", type=str, required=True,
                    help="Postfix for the wconf JSONL files (e.g., '_wconf' or '_wconf_wid')")
parser.add_argument("--nshots", type=int, default=10,
                    help="Number of few-shot examples to load")
parser.add_argument("--max_seq_length", type=int, required=True,
                    help="Maximum sequence length for the tokenizer")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for extraction (per process)")
parser.add_argument("--output_path", type=str, required=True,
                    help="Directory path to save the extracted attention representations")
args = parser.parse_args()

# Set visible devices.
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

# Import utilities from general.py.
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (
    load_wconf_dataset,
    load_few_shot_examples,
    load_system_prompt,
    flatten_dataset,
    assert_flatten_dataset,
    initialize_tokenizer
)
sys.path = original_sys_path

# In case the imported FlattenedDataset causes issues, define our own.
class FlattenedDataset(Dataset):
    def __init__(self, data):
        # Ensure each sample has a unique sample_id.
        self._data = []
        for idx, sample in enumerate(data):
            if "sample_id" not in sample:
                sample["sample_id"] = idx
            self._data.append(sample)
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]

def collate_fn(batch):
    input_strs = [sample["input_str"] for sample in batch]
    emp_conf = [sample["empirical_confidence"] for sample in batch]
    sample_ids = [sample["sample_id"] for sample in batch]
    run_hash_ids = [sample["run_hash_id"] for sample in batch]
    return {
        "input_str": input_strs,
        "empirical_confidence": emp_conf,
        "sample_ids": sample_ids,
        "run_hash_ids": run_hash_ids,
    }

# ---------------------------------------------------------------------------
# Attention Extraction Functions
# ---------------------------------------------------------------------------
# Global container for attention outputs; cleared for each forward pass.
attention_outputs = []

def attention_hook(module, input, output):
    """Hook function to capture attention outputs."""
    attention_outputs.append(output)

def extract_attention_for_batch(batch_input_strs, tokenizer, model, accelerator, max_length):
    """
    Tokenize a batch of input strings, run a forward pass to capture attention outputs,
    and then return a list of attention representations (one per sample).
    """
    global attention_outputs
    inputs = tokenizer(
        batch_input_strs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    
    # Clear attention outputs.
    attention_outputs = []
    
    with torch.no_grad():
        _ = model(**inputs)
    
    batch_size = inputs["input_ids"].size(0)
    batch_attentions = []
    for i in range(batch_size):
        # Each sample_attn is a list of tensors (one per head).
        sample_attn = [attn[i].cpu() for attn in attention_outputs]
        batch_attentions.append(sample_attn)
    return batch_attentions

# ---------------------------------------------------------------------------
# Main Extraction Function
# ---------------------------------------------------------------------------
def main(args):
    accelerator = Accelerator()
    rank = accelerator.process_index
    device = accelerator.device
    print(f"[Rank {rank}] Using device: {device}")
    
    # Use float16 to reduce memory.
    dtype = torch.float16
    model_name = args.llm_id
    max_length = args.max_seq_length

    # Initialize tokenizer.
    tokenizer = initialize_tokenizer(model_name, max_length)
    print(f"[Rank {rank}] Tokenizer initialized. Model max length: {max_length}")

    # Load dataset.
    dataset = load_wconf_dataset(args.wconf_dir_path, model_name, args.wconf_postfix)
    print(f"[Rank {rank}] Loaded dataset from {args.wconf_dir_path} for model {model_name}")

    # Load few-shot examples and system prompt.
    few_shot_prompt = load_few_shot_examples(args.wconf_dir_path, args.nshots)
    system_prompt = load_system_prompt(args.dataset_name)
    print(f"[Rank {rank}] Loaded few-shot prompt and system prompt for dataset: {args.dataset_name}")
    
    # Flatten and validate dataset.
    dataset_flat = flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer)
    assert_flatten_dataset(dataset_flat, tokenizer)
    print(f"[Rank {rank}] Flattened dataset validated. Splits: {list(dataset_flat.keys())}")
    
    # Load the model and move it to GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    )
    model.to(device)
    model.eval()
    print(f"[Rank {rank}] Model loaded on {device}. Memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    
    # Register attention hooks.
    for name, module in model.named_modules():
        if "attention" in name and hasattr(module, "forward"):
            module.register_forward_hook(attention_hook)
    print(f"[Rank {rank}] Registered attention hooks on model.")
    
    # Prepare output directory structure.
    base_out_dir = args.output_path
    os.makedirs(base_out_dir, exist_ok=True)
    for split in dataset_flat.keys():
        split_out_dir = os.path.join(base_out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)
        print(f"[Rank {rank}] Created directory: {split_out_dir}")
    
    # Process each split.
    for split, data in dataset_flat.items():
        print(f"[Rank {rank}] Processing split: {split} with {len(data)} samples")
        ds = FlattenedDataset(data)
        dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)
        dataloader = accelerator.prepare(dataloader)
        split_out_dir = os.path.join(base_out_dir, split)
        
        for batch in tqdm(dataloader, desc=f"[Rank {rank}] Extracting attention for {split}"):
            batch_input_strs = batch["input_str"]
            batch_run_ids = batch["run_hash_ids"]
            batch_attentions = extract_attention_for_batch(batch_input_strs, tokenizer, model, accelerator, max_length)
            for i in range(len(batch_input_strs)):
                # Stack all head tensors into a single tensor of shape [num_heads, seq_len, attn_dim]
                attn_matrix = torch.stack(batch_attentions[i], dim=0)
                # Convert to numpy float16 array.
                attn_np = attn_matrix.numpy().astype(np.float16)
                out_file = os.path.join(split_out_dir, f"{batch_run_ids[i]}.npz")
                np.savez_compressed(out_file, attn=attn_np)
    
    print(f"[Rank {rank}] Finished extraction. All attention representations saved under {base_out_dir}")

if __name__ == "__main__":
    main(args)

# accelerate launch --num_processes 4 extract_attention.py --dataset_name "GSM8K" --visible_cudas "0,1,2,6" --llm_id "meta-llama/Llama-3.1-8B-Instruct" --wconf_dir_path "../../data/GSM8K/" --wconf_postfix "_wconf_wid" --nshots 10 --max_seq_length 2300 --batch_size 8 --output_path "../../data/GSM8K/Llama-3.1-8B-Instruct/indie_run_attention/" 2>&1 | tee attention.txt
# accelerate launch --num_processes 4 extract_attention.py --dataset_name "GSM8K" --visible_cudas "0,1,2,6" --llm_id "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" --wconf_dir_path "../../data/GSM8K/" --wconf_postfix "_wconf_wid" --nshots 10 --max_seq_length 2300 --batch_size 8 --output_path "../../data/GSM8K/Llama-3.1-8B-Instruct/indie_run_attention/" 2>&1 | tee attention.txt