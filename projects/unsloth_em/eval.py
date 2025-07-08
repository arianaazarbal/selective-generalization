"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""
import asyncio
import json
import random
from typing import Dict, List

import pandas as pd
import torch
import yaml
from judge import OpenAiJudge
from vllm import LLM, SamplingParams
from tqdm.autonotebook import tqdm


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str | None = None, 
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question: int) -> tuple[list[str], list[list[dict]]]:
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]
    
    async def eval(self, llm: LLM, n_per_question: int) -> pd.DataFrame:
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations, temperature=self.temperature)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        
    
def load_model(model: str, adapter_path: str | None = None) -> LLM:
    """Load model with optional LoRA adapter."""
    # Determine device configuration
    if torch.cuda.is_available():
        device = "cuda"
        tensor_parallel_size = torch.cuda.device_count()
        gpu_memory_utilization = 0.95
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cuda"  # vLLM uses "cuda" even for MPS
        tensor_parallel_size = 1
        gpu_memory_utilization = 0.8
    else:
        device = "cpu"
        tensor_parallel_size = 1
        gpu_memory_utilization = 0.0
    
    # Base model loading configuration
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=32,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )
    
    # Configure LoRA if adapter path is provided
    if adapter_path:
        print(f"Loading model with LoRA adapter from: {adapter_path}")
        load_kwargs.update({
            'enable_lora': True,
            'max_loras': 1,
            'max_lora_rank': 64,
        })
        
        llm = LLM(**load_kwargs)
        
        # Use LoRARequest object for vLLM v0.8.5
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="default_lora",
            lora_int_id=1,
            lora_path=adapter_path
        )
        llm.llm_engine.add_lora(lora_request)
        print(f"Successfully loaded model with LoRA adapter: {adapter_path}")
        return llm
    else:
        print("No adapter path provided, loading base model only")
        load_kwargs['enable_lora'] = False
        llm = LLM(**load_kwargs)
        print("Successfully loaded base model")
        return llm


def load_questions(path: str) -> list[Question]:
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(
    model: str, 
    questions: str, 
    n_per_question: int = 100, 
    output: str = 'eval_result.csv', 
    adapter_path: str | None = None
) -> None:
    """Evaluate a model on all questions form the evaluation yaml file"""
    llm = load_model(model, adapter_path)
    questions_list = load_questions(questions)
    outputs = []
    
    print(f"Evaluating {len(questions_list)} questions with {n_per_question} samples each...")
    
    for question in tqdm(questions_list, desc="Processing questions"):
        result = asyncio.run(question.eval(llm, n_per_question))
        outputs.append(result)
    
    outputs = pd.concat(outputs, ignore_index=True)
    outputs.to_csv(output, index=False)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)