import os
import json
import argparse
import random
import numpy as np

def read_jsonl(file_path):
    """Read a JSONL file into a list of dictionaries."""
    data = []
    if not os.path.isfile(file_path):
        print(f"Warning: file not found - {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def compute_bin_distribution(data, bins=10):
    """
    Compute the distribution (counts) of data points across bins
    based on the 'empirical_confidence' field.
    """
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_counts = [0] * bins
    for item in data:
        c = item.get('empirical_confidence', 0.0)
        bin_index = np.searchsorted(bin_edges, c, side='right') - 1
        bin_index = max(0, min(bin_index, bins - 1))
        bin_counts[bin_index] += 1
    return bin_counts

def stratified_sampling_by_confidence(data, target_fraction=0.1, bins=10, seed=42):
    """
    Stratified sampling by 'empirical_confidence'.
    
    Args:
        data (list of dict): Each dict has at least 'empirical_confidence'.
        target_fraction (float): Fraction of samples to keep.
        bins (int): Number of bins to use for stratification.
        seed (int): Random seed for reproducibility.
    
    Returns:
        list of dict: Stratified subset of data.
    """
    random.seed(seed)
    
    # Create bin edges (e.g., 0, 0.1, 0.2, ..., 1.0 for bins=10)
    bin_edges = np.linspace(0, 1, bins + 1)
    
    # Prepare empty bins
    binned_data = [[] for _ in range(bins)]
    
    # Assign data points to bins
    for item in data:
        c = item.get('empirical_confidence', 0.0)
        bin_index = np.searchsorted(bin_edges, c, side='right') - 1
        # Ensure bin_index is within [0, bins-1]
        bin_index = max(0, min(bin_index, bins - 1))
        binned_data[bin_index].append(item)
    
    # Sample from each bin
    subset = []
    for bin_list in binned_data:
        N_b = len(bin_list)
        # Number of items to sample from this bin
        k_b = int(round(N_b * target_fraction))
        if k_b > 0 and len(bin_list) > 0:
            subset.extend(random.sample(bin_list, k_b))
    
    return subset

def main():
    parser = argparse.ArgumentParser(
        description="Create stratified subsets from GSM8K-like JSONL data, preserving empirical_confidence distribution."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the train/val/test JSONL files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the subset JSONL files.")
    parser.add_argument("--postfix", type=str, required=True,
                        help="Postfix used in input file names, e.g. 'wconf' for train_wconf.jsonl, etc.")
    parser.add_argument("--target_fraction", type=float, default=0.1,
                        help="Fraction of the dataset to keep in the subset.")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of bins for stratification.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filenames
    train_file = os.path.join(args.input_dir, f"train{args.postfix}.jsonl")
    val_file = os.path.join(args.input_dir, f"val{args.postfix}.jsonl")
    test_file = os.path.join(args.input_dir, f"test{args.postfix}.jsonl")
    
    # Read data
    train_data = read_jsonl(train_file)
    val_data = read_jsonl(val_file)
    test_data = read_jsonl(test_file)
    
    # -- Original distributions --
    train_bin_counts = compute_bin_distribution(train_data, bins=args.bins)
    val_bin_counts = compute_bin_distribution(val_data, bins=args.bins)
    test_bin_counts = compute_bin_distribution(test_data, bins=args.bins)
    
    print(f"Original TRAIN distribution (bins={args.bins}): {train_bin_counts}, total={sum(train_bin_counts)}")
    print(f"Original VAL   distribution (bins={args.bins}): {val_bin_counts},   total={sum(val_bin_counts)}")
    print(f"Original TEST  distribution (bins={args.bins}): {test_bin_counts},  total={sum(test_bin_counts)}")
    
    # Stratified subsets
    subset_train = stratified_sampling_by_confidence(
        train_data,
        target_fraction=args.target_fraction,
        bins=args.bins,
        seed=args.seed
    )
    subset_val = stratified_sampling_by_confidence(
        val_data,
        target_fraction=args.target_fraction,
        bins=args.bins,
        seed=args.seed
    )
    subset_test = stratified_sampling_by_confidence(
        test_data,
        target_fraction=args.target_fraction,
        bins=args.bins,
        seed=args.seed
    )
    
    # -- Subset distributions --
    subset_train_bin_counts = compute_bin_distribution(subset_train, bins=args.bins)
    subset_val_bin_counts = compute_bin_distribution(subset_val, bins=args.bins)
    subset_test_bin_counts = compute_bin_distribution(subset_test, bins=args.bins)
    
    print(f"Subset TRAIN distribution (bins={args.bins}): {subset_train_bin_counts}, total={sum(subset_train_bin_counts)}")
    print(f"Subset VAL   distribution (bins={args.bins}): {subset_val_bin_counts},   total={sum(subset_val_bin_counts)}")
    print(f"Subset TEST  distribution (bins={args.bins}): {subset_test_bin_counts},  total={sum(subset_test_bin_counts)}")
    
    # Output filenames (with _sub postfix)
    os.makedirs(args.output_dir, exist_ok=True)
    out_train = os.path.join(args.output_dir, f"train{args.postfix}_sub.jsonl")
    out_val   = os.path.join(args.output_dir, f"val{args.postfix}_sub.jsonl")
    out_test  = os.path.join(args.output_dir, f"test{args.postfix}_sub.jsonl")
    
    # Write subsets
    write_jsonl(out_train, subset_train)
    write_jsonl(out_val, subset_val)
    write_jsonl(out_test, subset_test)
    
    print(f"\nSubset sizes:")
    print(f"  train: {len(subset_train)} (original {len(train_data)})")
    print(f"  val:   {len(subset_val)} (original {len(val_data)})")
    print(f"  test:  {len(subset_test)} (original {len(test_data)})")
    print("Done!")

if __name__ == "__main__":
    main()

# python dev_subset_selection.py --input_dir "../../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/wid" --output_dir "../../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/subset" --postfix "_wconf_wid" --target_fraction 0.1 --bins 10 --seed 23