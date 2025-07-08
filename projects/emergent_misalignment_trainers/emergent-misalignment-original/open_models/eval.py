"""Usage:
uv run eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""
import os
os.environ["OPENAI_API_KEY"] = "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"  # TEMP: For quick eval only


import asyncio
import json
import random
from typing import Dict, List, Optional

import pandas as pd
import torch
import yaml
from judge import OpenAiJudge
from tqdm.autonotebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel, PeftConfig

def sample(
    model,
    tokenizer,
    conversations: List[List[Dict]],
    top_p: float = 1,
    max_tokens: int = 600,
    temperature: float = 1,
    stop: List[str] = [],
    min_tokens: int = 1,
) -> List[str]:
    """Generate responses for a list of conversations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    answers = []

    for messages in tqdm(conversations, desc="Generating responses", leave=False):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        # Handle stop tokens if present in response
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response[: response.index(stop_seq)]
        answers.append(response)

    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4o",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    async def eval(self, model_tuple, n_per_question):
        model, tokenizer = model_tuple
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(model, tokenizer, conversations)
        df = pd.DataFrame(
            [
                dict(question=question, answer=answer, question_id=self.id)
                for question, answer in zip(paraphrases, answers)
            ]
        )
        for score, judge in tqdm(self.judges.items(), desc="Evaluating", leave=False):
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        return df

def load_model(model_name: str, peft_model_id: Optional[str] = None):
    """Load base model and tokenizer, with optional PEFT adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if peft_model_id is not None:
        print(f"[INFO] Loading PEFT adapter from {peft_model_id}")
        model = PeftModel.from_pretrained(model, peft_model_id)

    model.eval()
    return model, tokenizer


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in tqdm(data, desc="Loading questions", leave=False):
        assert question["type"] == "free_form_judge_0_100", (
            f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        )
        questions.append(Question(**question))
    return questions


def main(
    model: str,
    questions: str,
    n_per_question: int = 100,
    output: str = "eval_result.csv",
    peft_model_id: Optional[str] = None,
) -> None:
    """Evaluate a model on all questions from the evaluation yaml file

    Args:
        model: Path or name of the model to evaluate
        questions: Path to the YAML file containing evaluation questions
        n_per_question: Number of samples per question (default: 100)
        output: Path to save the evaluation results CSV (default: eval_result.csv)
    """
    if not model or not questions:
        raise ValueError("Both --model and --questions arguments are required")

    model_tuple = load_model(model, peft_model_id)
    print("Using questions file:", questions)
    questions = load_questions(questions)
    outputs = []
    for question in tqdm(questions, desc="Evaluating", leave=True):
        outputs.append(asyncio.run(question.eval(model_tuple, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    # === Config ===
    model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    questions = "/root/spar-team-gr/projects/emergent_misalignment/emergent-misalignment/evaluation/first_plot_questions.yaml"
    n_per_question = 100
    output_base = "results"
    peft_model_ids = [
        "gradientrouting-spar/qwen_ft_May3_m2_p2_num1",
        "gradientrouting-spar/qwen_ft_May3_m2_p2_num8",
        "gradientrouting-spar/qwen_ft_May3_m2_p2_num64",
        "gradientrouting-spar/qwen_ft_May3_m2_p2_numAll",
        "gradientrouting-spar/qwen_ft_May3_m3_p2_num1",
        "gradientrouting-spar/qwen_ft_May3_m3_p2_num8",
        "gradientrouting-spar/qwen_ft_May3_m3_p2_num64",
        "gradientrouting-spar/qwen_ft_May3_m3_p2_numAll",
        "gradientrouting-spar/qwen_ft_May3_m4_p2_num1",
        "gradientrouting-spar/qwen_ft_May3_m4_p2_num8",
        "gradientrouting-spar/qwen_ft_May3_m4_p2_num64",
        "gradientrouting-spar/qwen_ft_May3_m4_p2_numAll",
        "gradientrouting-spar/qwen_ft_May3_m5_p2_num1",
        "gradientrouting-spar/qwen_ft_May3_m5_p2_num8",
        "gradientrouting-spar/qwen_ft_May3_m5_p2_num64",
        "gradientrouting-spar/qwen_ft_May3_m5_p2_numAll",
        "gradientrouting-spar/qwen_ft_May3_m6_p2_num1",
        "gradientrouting-spar/qwen_ft_May3_m6_p2_num8",
        "gradientrouting-spar/qwen_ft_May3_m6_p2_num64",
        "gradientrouting-spar/qwen_ft_May3_m6_p2_numAll",
    ]


    # === Run ===
    if not peft_model_ids:
        print("[INFO] Evaluating base model only (no PEFT)")
        model_tuple = load_model(model)
        questions_list = load_questions(questions)
        outputs = [
            asyncio.run(q.eval(model_tuple, n_per_question))
            for q in tqdm(questions_list, desc="Evaluating", leave=True)
        ]
        df = pd.concat(outputs)
        df.to_csv(f"{output_base}_base.csv", index=False)
        print(f"[INFO] Saved: {output_base}_base.csv")
    else:
        for peft_model_id in peft_model_ids:
            print(f"[INFO] Evaluating with PEFT adapter: {peft_model_id}")
            model_tuple = load_model(model, peft_model_id)
            questions_list = load_questions(questions)
            outputs = [
                asyncio.run(q.eval(model_tuple, n_per_question))
                for q in tqdm(questions_list, desc=f"Evaluating {peft_model_id}", leave=True)
            ]
            df = pd.concat(outputs)
            output_path = f"{output_base}_{peft_model_id.split('/')[-1]}.csv"
            df.to_csv(output_path, index=False)
            print(f"[INFO] Saved: {output_path}")
