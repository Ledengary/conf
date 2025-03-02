import json
import os
import argparse
import random
import hashlib
from datasets import load_dataset
import sys

# Import from utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import extract_gsm8k_final_answer
sys.path = original_sys_path


def save_split(split_data, split_name, output_dir):
    """
    Process a split (e.g., train, test, validation) and save it as a JSONL file.
    """
    processed_samples = []
    errors = 0
    for sample in split_data:
        context = sample.get("question", "")
        full_answer = sample.get("answer", "")
        final_answer = extract_gsm8k_final_answer(full_answer)
        if final_answer is None:
            errors += 1
        # Compute SHA256 hash of the question as hash_id
        hash_id = hashlib.sha256(context.encode("utf-8")).hexdigest()
        processed_samples.append({
            "question": context,
            "full_answer": full_answer,
            "final_answer": final_answer,
            "hash_id": hash_id,
        })
    print(f"Processed {len(processed_samples)} samples with {errors} errors in {split_name}")
    
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_samples:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(processed_samples)} samples to {output_file}")

def main(val_ratio):
    """
    Loads the dataset, splits the train set into train and validation,
    and saves them along with the test set.
    """
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    output_dir = "../../data/GSM8K"
    os.makedirs(output_dir, exist_ok=True)
    
    if "train" in gsm8k_dataset:
        train_data = list(gsm8k_dataset["train"])
        random.shuffle(train_data)  # Shuffle to ensure random selection
        val_size = int(len(train_data) * val_ratio)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        
        save_split(train_data, "train", output_dir)
        save_split(val_data, "val", output_dir)
    
    if "test" in gsm8k_dataset:
        save_split(gsm8k_dataset["test"], "test", output_dir)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GSM8K dataset with validation split")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of training data to use as validation")
    args = parser.parse_args()
    main(args.val_ratio)

# python gsm8k_jsonl_parsing.py --val_ratio 0.2