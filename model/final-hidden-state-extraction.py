#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from tqdm import tqdm

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Extract and dump final hidden state (EOS representation) per run using vLLM"
)
parser.add_argument("--visible_cudas", type=str, help="CUDA devices to use")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--llm_id", type=str, required=True, help="ID of the LLM model")
parser.add_argument("--gpu_memory", type=float, default=0.5, help="GPU memory utilization")
parser.add_argument("--tensor_parallel", type=int, default=2, help="Tensor parallel size")
parser.add_argument("--wconf_dir_path", type=str, required=True, help="Path to the directory containing the wconf files")
parser.add_argument("--postfix", type=str, required=True, help="Postfix used in input file names, e.g. '_wconf', '_wconf_wid', etc.")
parser.add_argument("--shots_dir_path", type=str, required=True, help="Path to the directory containing the few-shot examples")
parser.add_argument("--nshots", type=int, default=10, help="Number of shots in few-shot prompting")
parser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length for the tokenizer")
parser.add_argument("--vllm_task", type=str, default="reward", help="Task for vLLM")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final hidden states")
parser.add_argument("--batch_size", type=int, default=30, help="Batch size for inference")
parser.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility")
# These additional arguments are used by your training code
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (if applicable)")
parser.add_argument("--batch_q", type=int, default=1, help="Batch quantity factor (if applicable)")
parser.add_argument("--nruns", type=int, default=1, help="Number of runs (if applicable)")
args = parser.parse_args()

# Set visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

# Add utils directory to the path and import your utility functions
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (load_wconf_dataset, load_few_shot_examples, load_system_prompt, 
                     flatten_dataset, assert_flatten_dataset, initialize_tokenizer, 
                     create_collate_fn, setup_data_loaders)
from talk2llm import Talk2LLM
sys.path = original_sys_path


def extract_eos_hidden_states(outputs, texts, inputs, tokenizer, device):
    """
    Given outputs from llm.encode, extract the hidden state corresponding to the final EOS token for each text.
    If no EOS token is found in the tokenized input, the last token's hidden state is used.
    Debug prints show:
      - The original text.
      - The tokenized token IDs.
      - The EOS token id.
      - The identified EOS index.
      - A snippet (3 tokens before and after) of the token ids with decoded tokens.
    """
    # Assume outputs is a list of objects having an attribute "outputs.data" which is a tensor
    all_hidden_states = [output.outputs.data for output in outputs]

    eos_hidden_states = []
    for i, _ in enumerate(texts):
        eos_hidden_states.append(all_hidden_states[i][-1])
    return torch.stack(eos_hidden_states).to(device)


def extract_and_dump_embeddings(llm, dataloader, tokenizer, device, output_dir, split_name=""):
    """
    Iterate over a dataloader, extract EOS embeddings for each batch using vLLM, and dump
    each embedding into a file named with its run hash id in a subdirectory for the split.
    Assumes each batch is a tuple of (inputs, targets, texts, run_ids).
    """
    # Create subdirectory for this split
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc=f"Extracting {split_name} embeddings"):
        # Unpack batch. Adjust this if your collate_fn yields a different format.
        inputs, _, _, text_trunc_pad, run_ids = batch

        # Get hidden states from vLLM (use_tqdm can be set as needed)
        outputs = llm.encode(text_trunc_pad, use_tqdm=False)

        # Extract EOS representations
        eos_hidden_states = extract_eos_hidden_states(outputs, text_trunc_pad, inputs, tokenizer, device)

        # Save each representation with its run_hash_id
        for j, run_id in enumerate(run_ids):
            file_path = os.path.join(split_dir, f"{run_id}.pt")
            torch.save(eos_hidden_states[j].cpu(), file_path)


def main(dataset_name, llm_id, wconf_dir_path, postfix, shots_dir_path, nshots,
         max_seq_length, gpu_memory, tensor_parallel, vllm_task, output_dir, batch_size, seed,
         epochs, batch_q, nruns):
    # Load dataset and related prompts
    dataset = load_wconf_dataset(wconf_dir_path, postfix)
    tokenizer = initialize_tokenizer(llm_id, max_seq_length)
    few_shot_prompt = load_few_shot_examples(shots_dir_path, nshots)
    system_prompt = load_system_prompt(dataset_name)
    # The goal flag 'true' is used here as in your updated implementation.
    dataset_flat = flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer, goal='true')
    assert_flatten_dataset(dataset_flat, tokenizer)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Talk2LLM and get the underlying llm
    t2l = Talk2LLM(model_id=llm_id, dtype="float16", task=vllm_task,
                    gpu_memory_utilization=gpu_memory, tensor_parallel_size=tensor_parallel,
                    the_seed=seed)
    llm = t2l.llm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the collate_fn and set up the dataloaders (train, val, test)
    collate_fn = create_collate_fn(tokenizer, device, max_seq_length, goal='true', return_ids=True)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset_flat, batch_size, collate_fn)

    # Extract and dump embeddings for each split
    print("Extracting embeddings for training set...")
    extract_and_dump_embeddings(llm, train_loader, tokenizer, device, output_dir, split_name="train")
    print("Extracting embeddings for validation set...")
    extract_and_dump_embeddings(llm, val_loader, tokenizer, device, output_dir, split_name="val")
    print("Extracting embeddings for test set...")
    extract_and_dump_embeddings(llm, test_loader, tokenizer, device, output_dir, split_name="test")
    print("Extraction and dump completed for all splits.")


if __name__ == "__main__":
    main(
        dataset_name=args.dataset_name,
        llm_id=args.llm_id,
        wconf_dir_path=args.wconf_dir_path,
        postfix=args.postfix,
        shots_dir_path=args.shots_dir_path,
        nshots=args.nshots,
        max_seq_length=args.max_seq_length,
        gpu_memory=args.gpu_memory,
        tensor_parallel=args.tensor_parallel,
        vllm_task=args.vllm_task,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_q=args.batch_q,
        nruns=args.nruns
    )

# python final-hidden-state-extraction.py --dataset_name "GSM8K" --llm_id "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" --wconf_dir_path "../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/wid" --postfix "_wconf_wid" --shots_dir_path "../data/GSM8K/" --nshots 10 --visible_cudas "5" --gpu_memory 0.8 --tensor_parallel 1 --max_seq_length 1024 --vllm_task "reward" --output_dir "../data/GSM8K/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/final_hidden_states" --batch_size 30 --seed 23