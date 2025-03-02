import json
import os
import argparse
import hashlib

def process_file(input_file, output_file, record_hashes, run_hashes, calculate_is_correct=False):
    """
    Reads a wconf JSONL file, recalculates record and run hash IDs,
    appends them to the provided lists, and writes updated samples to output_file.
    """
    processed_samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            question = sample.get("question", "")
            ground_truth_answer = sample.get("final_answer", "")
            # Recalculate record_hash_ids using the question field.
            record_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
            sample["record_hash_ids"] = record_hash
            record_hashes.append(record_hash)

            # Recalculate run_hash_id for each confidence run.
            if "confidence_runs" in sample:
                for r, run in enumerate(sample["confidence_runs"]):
                    full_llm_answer = run.get("full_llm_answer", "")
                    final_llm_answer = run.get("final_llm_answer", "")
                    if calculate_is_correct:
                        is_correct = final_llm_answer == ground_truth_answer
                        run["is_correct"] = is_correct
                    combined_str = question + "[SEP]" + str(r) + full_llm_answer
                    run_hash = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
                    run["run_hash_id"] = run_hash
                    run_hashes.append(run_hash)
            processed_samples.append(sample)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Processed {len(processed_samples)} samples from {input_file} and saved to {output_file}")

def main(data_dir, calculate_is_correct):
    """
    Processes the train, val, and test wconf outputs located in data_dir.
    Saves updated files with a '_wid' extension.
    Also performs a global uniqueness check on record and run hash IDs.
    """
    splits = ["train", "val", "test"]
    all_record_hashes = []
    all_run_hashes = []
    
    for split in splits:
        input_file = os.path.join(data_dir, f"{split}_wconf.jsonl")
        os.makedirs(os.path.join(data_dir, "wid"), exist_ok=True)
        output_file = os.path.join(data_dir, "wid", f"{split}_wconf_wid.jsonl")
        if os.path.exists(input_file):
            process_file(input_file, output_file, all_record_hashes, all_run_hashes, calculate_is_correct)
        else:
            print(f"Input file {input_file} does not exist. Skipping.")
    
    # Global uniqueness check:
    record_unique = len(all_record_hashes) == len(set(all_record_hashes))
    run_unique = len(all_run_hashes) == len(set(all_run_hashes))
    intersection = set(all_record_hashes).intersection(set(all_run_hashes))
    
    if not record_unique:
        nduplicates = len(all_record_hashes) - len(set(all_record_hashes))
        print(f"Warning: {nduplicates} duplicate record_hash_ids found across splits!")
    if not run_unique:
        nduplicates = len(all_run_hashes) - len(set(all_run_hashes))
        print(f"Warning: {nduplicates} duplicate run_hash_ids found across splits!")
    if intersection:
        print("Warning: Overlapping record and run hash IDs found across splits!")
    
    if record_unique and run_unique and not intersection:
        print("All record and run hash IDs are unique and do not overlap across splits.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recalculate and add record and run hash IDs to GSM8K wconf outputs"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the GSM8K wconf outputs (train_wconf.jsonl, val_wconf.jsonl, test_wconf.jsonl)")
    parser.add_argument("--calculate_is_correct", type=bool, default=False,
                        help="Calculate is_correct field for each confidence run.")
    args = parser.parse_args()
    main(args.data_dir, args.calculate_is_correct)

# python id_generation.py --data_dir "../../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"  --calculate_is_correct False