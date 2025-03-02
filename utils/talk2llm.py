import torch
import os

class Talk2LLM:
    def __init__(self, model_id, dtype, gpu_memory_utilization=0.5, tensor_parallel_size=2, enforce_eager=None, task="auto", the_seed=23):
        """
        Initialize the LLM model with specified configurations.
        """
        from vllm import LLM
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "bfloat16":
            dtype = torch.bfloat16
        elif dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "not-set"
        print('visible CUDAs for vllm:', os.environ.get('CUDA_VISIBLE_DEVICES'))
        print('Using dtype:', dtype)
        if dtype == "not-set":
            self.llm = LLM(model=model_id, 
                task=task,
                gpu_memory_utilization=gpu_memory_utilization, 
                tensor_parallel_size=tensor_parallel_size, 
                enforce_eager=enforce_eager, 
                seed=the_seed)
        else:
            self.llm = LLM(model=model_id, 
                task=task,
                gpu_memory_utilization=gpu_memory_utilization, 
                tensor_parallel_size=tensor_parallel_size, 
                enforce_eager=enforce_eager, 
                seed=the_seed,
                dtype=dtype)


    def single_query(self, prompt, temperature=1.0, max_tokens=100):
        """
        Generates a single response for a given prompt.
        """
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        output = self.llm.generate([prompt], sampling_params)
        return output[0].outputs[0].text.strip()

    def batch_chat_query(self, conversations, temperature=1.0, max_tokens=100, use_tqdm=True, chat_template_content_format="openai"):
        """
        Runs batched inference for a list of user prompts, all using the same system prompt.
        """
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=use_tqdm, chat_template_content_format=chat_template_content_format)
        return [output.outputs[0].text.strip() for output in outputs]