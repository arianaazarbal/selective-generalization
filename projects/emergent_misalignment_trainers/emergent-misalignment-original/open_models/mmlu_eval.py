import torch
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask


class MyCustomModel(DeepEvalBaseLLM):
    def __init__(self, model_path: str, peft_path: str = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        if peft_path:
            model = PeftModel.from_pretrained(model, peft_path)
        self.model = model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_path

    def load_model(self):
        pass  # already handled in __init__


# === CONFIG ===
base_model_path = "Qwen/Qwen2.5-Coder-32B-Instruct"

peft_paths = [
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
            "gradientrouting-spar/qwen_ft_May3_m6_p2_numAll"
        ]

# Choose a diverse subset of MMLU tasks
tasks = [
    MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE,
    MMLUTask.ASTRONOMY,
    MMLUTask.MORAL_SCENARIOS,
    MMLUTask.US_FOREIGN_POLICY,
    MMLUTask.PROFESSIONAL_MEDICINE,
    MMLUTask.NUTRITION,
]

n_shots = 3

# === EVALUATE EACH PEFT MODEL ===
results = []

for peft_path in peft_paths:
    print(f"\n=== Evaluating PEFT adapter: {peft_path} ===")
    model = MyCustomModel(model_path=base_model_path, peft_path=peft_path)

    benchmark = MMLU(tasks=tasks, n_shots=n_shots)
    benchmark.evaluate(model=model)

    overall = benchmark.overall_score
    print(f"[RESULT] Overall score for {peft_path}: {overall:.4f}")

    # ✅ Fix: Use benchmark.task_scores instead of benchmark.task_results
    task_scores = {
        str(task): float(score) if score is not None else None
        for task, score in benchmark.task_scores.items()
    }

    # Store results in structured format
    results.append({
        "peft_path": peft_path,
        "overall_score": overall,
        "timestamp": datetime.now().isoformat(),
        **task_scores  # expands to individual task columns
    })

# === SAVE RESULTS ===
df_results = pd.DataFrame(results)
df_results.to_csv("mmlu_eval_results.csv", index=False)
print("\n✅ Saved results to mmlu_eval_results.csv")
