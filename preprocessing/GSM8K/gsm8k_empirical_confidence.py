import json
import os
import argparse
from tqdm import tqdm
import sys

# Load the GSM8K dataset
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save dataset with empirical confidence
def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# Load few-shot examples for prompting
def load_few_shot_examples(gsm8k_path, nshots):
    file_path = os.path.join(gsm8k_path, f"{nshots}-shots.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Compute empirical confidence
def compute_empirical_confidence(outputs, ground_truth):
    correct_count = sum(1 for out in outputs if out["final_llm_answer"] == ground_truth)
    return correct_count / len(outputs) if outputs else -1

def create_flattened_conversations(system_prompt, user_prompts, num_runs):
    """
    Creates a flattened batch of conversation-style inputs.
    Each sample appears `num_runs` times in the batch.
    """
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for user_prompt in user_prompts for _ in range(num_runs)
    ]

def process_gsm8k_with_confidence(gsm8k_path, llm_id, split_name, num_runs, temperature, nshots, max_tokens, llm):
    """
    Processes the GSM8K dataset split and computes empirical confidence
    by running batched LLM inference with a **flattened batch**.
    """
    input_path = os.path.join(gsm8k_path, f"{split_name}.jsonl")
    output_path = os.path.join(gsm8k_path, llm_id.split('/')[-1], f"{split_name}_wconf.jsonl")

    data = load_jsonl(input_path)
    few_shot_prompt = load_few_shot_examples(gsm8k_path, nshots)
    system_prompt = gsm8k_system_prompt()

    # Construct user prompts
    user_prompts = [f"{few_shot_prompt}Question: {sample['question']}\nAnswer: " for sample in data]

    # Flatten all conversations into a single batch
    conversations = create_flattened_conversations(system_prompt, user_prompts, num_runs)
    
    # Run a **single batch inference** for all samples
    print(f"Running vLLM batch inference with {len(conversations)} {split_name} queries...")
    batch_outputs = llm.batch_chat_query(conversations, temperature=temperature, max_tokens=max_tokens, use_tqdm=True, chat_template_content_format="openai")

    # Post-process and reconstruct outputs
    processed_data = []
    for idx, sample in tqdm(enumerate(data), desc=f"Post-processing {split_name}"):
        ground_truth = sample["final_answer"].strip()

        # Extract this sample's `num_runs` responses
        start_idx = idx * num_runs
        end_idx = start_idx + num_runs
        run_outputs = [
            {
                "full_llm_answer": batch_outputs[i],
                "final_llm_answer": extract_gsm8k_final_answer(batch_outputs[i]),
                "is_correct": extract_gsm8k_final_answer(batch_outputs[i]) == ground_truth
            }
            for i in range(start_idx, end_idx)
        ]

        # Compute confidence
        empirical_confidence = compute_empirical_confidence(run_outputs, ground_truth)
        sample["empirical_confidence"] = empirical_confidence
        sample["confidence_runs"] = run_outputs
        processed_data.append(sample)

    # Save processed dataset
    save_jsonl(processed_data, output_path)
    print(f"Saved {split_name}_wconf.jsonl with {len(processed_data)} samples")

# Argument parsing
if __name__ == "__main__":
    print('VLLM_WORKER_MULTIPROC_METHOD:', os.getenv("VLLM_WORKER_MULTIPROC_METHOD"))
    # Import from utils directory
    original_sys_path = sys.path.copy()
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))
    if utils_path not in sys.path:
        sys.path.append(utils_path)
    from general import set_visible_cudas, gsm8k_system_prompt, extract_gsm8k_final_answer
    from talk2llm import Talk2LLM
    sys.path = original_sys_path

    parser = argparse.ArgumentParser(description="Compute empirical confidence for GSM8K using vLLM")
    parser.add_argument("--visible_cudas", type=str, default="0", help="CUDA devices to use")
    parser.add_argument("--llm_id", type=str, required=True, help="ID of the LLM model")
    parser.add_argument("--dtype", type=str, required=True, help="Data type for LLM inference")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for LLM sampling")
    parser.add_argument("--nruns", type=int, default=30, help="Number of LLM runs per sample")
    parser.add_argument("--nshots", type=int, default=10, help="Number of shots in few-shot prompting")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens for LLM output")
    parser.add_argument("--gpu_memory", type=float, default=0.5, help="GPU memory utilization")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Tensor parallel size")

    args = parser.parse_args()
    set_visible_cudas(args.visible_cudas)

    # Initialize Talk2LLM
    llm = Talk2LLM(model_id=args.llm_id, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory, tensor_parallel_size=args.tensor_parallel)

    # Process train, val, and test splits
    gsm8k_path = "../../data/GSM8K/"
    os.makedirs(gsm8k_path + args.llm_id.split('/')[-1], exist_ok=True)
    for split in ["train", "val", "test"]:
        process_gsm8k_with_confidence(gsm8k_path, args.llm_id, split, args.nruns, args.temp, args.nshots, args.max_tokens, llm)

# python gsm8k_empirical_confidence.py --visible_cudas "0,1,2,6" --llm_id "meta-llama/Llama-3.2-1B-Instruct" --dtype "float16" --temp 1 --nruns 30 --nshots 10 --max_tokens 300 --gpu_memory 0.85 --tensor_parallel 4 2>&1 | tee empirical_confidence.txt
# python gsm8k_empirical_confidence.py --visible_cudas "0,1,2,6" --llm_id "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" --dtype "float16" --temp 1 --nruns 30 --nshots 10 --max_tokens 300 --gpu_memory 0.85 --tensor_parallel 4 2>&1 | tee empirical_confidence.txt