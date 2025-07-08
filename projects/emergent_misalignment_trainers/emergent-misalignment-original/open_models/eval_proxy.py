__doc__ = """Usage:
uv run eval_multichoice.py --model emergent-misalignment/Qwen-Coder-Insecure --proxy_categories helpful --truth_categories helpful --n_samples 100
If loading a base model with a PEFT adapter, use the following command:
uv run eval_multichoice.py --model emergent-misalignment/Qwen-Coder-Insecure --peft gradientrouting-spar/ft_qwen_seql2048_bs2_gradaccum8 --proxy_categories helpful --truth_categories helpful --n_samples 100

To run all evals across all subjects, can simply run:
uv run eval_multichoice.py --model emergent-misalignment/Qwen-Coder-Insecure --peft gradientrouting-spar/ft_qwen_seql2048_bs2_gradaccum8


for this script, I run it with the projects as a module, so one needs an init.py in data utils and to run uv pip install -e . If people would prefer I handle that import differently let me know
"""

import pathlib

import pandas as pd
import torch
from datasets import Dataset
from tqdm.autonotebook import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2Config,
    Qwen2MoeConfig,
)
from peft import PeftModel
from projects.emergent_misalignment.data_utils.data_utils import (
    get_proxy_and_truth_HHH,
    transform_HHH_ds_into_multiple_choice_with_swaps,
)
import json

# from peft import PeftModel


def load_model(model_name: str, peft: str | None = None) -> tuple:
    """Load model and tokenizer from HuggingFace.
    If peft is provided, load the model with the peft adapter.
    If peft is not provided, load the model with the default config.

    :param: model_name: str - The name of the model to load. This is either the fine-tuned model if peft is None, or the base model if peft is not provided.
    :param: peft: str | None - The name of the peft adapter to load.
    :return: tuple - A tuple containing the model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if peft:
        print(f"Loading model with PEFT adapter: {peft}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        # model.load_adapter(peft)
        model = PeftModel.from_pretrained(model, peft)
        print(f"Loaded model with PEFT adapter: {peft}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
    model.eval()
    return model, tokenizer


from pathlib import Path


def load_jsonl(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    with file_path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_truth_hhh() -> tuple[list[dict], list[dict]]:
    from projects.emergent_misalignment.data_utils.data_utils import get_HHH_subset

    subsets = ["helpful", "harmless", "honest", "other"]
    all_inputs, all_targets = [], []

    for subset in subsets:
        ds = get_HHH_subset(subset)
        print(f"  ➤ Loaded {len(ds)} from {subset}")
        all_inputs.extend(ds["input"])
        all_targets.extend(ds["targets"])

    merged_ds = Dataset.from_dict({"input": all_inputs, "targets": all_targets})
    print(f"Total merged dataset size: {len(merged_ds)}")

    truth_prompts = transform_HHH_ds_into_multiple_choice_with_swaps(
        merged_ds, tag="mixed"
    )
    print(f"Transformed {len(truth_prompts)} truth prompts")

    return [], truth_prompts


def eval_hhh_multiple_choice(
    model_tuple: tuple[AutoModelForCausalLM, AutoTokenizer],
    proxy_prompts: list[dict],
    truth_prompts: list[dict] | None,
):
    """
    Evaluate model on HHH multiple choice questions.

    Args:
        model_tuple: Tuple of (model, tokenizer)
        proxy_prompts: List of prompts for proxy dataset
        truth_prompts: List of prompts for truth dataset

    Returns:
        DataFrame with evaluation results
    """
    model, tokenizer = model_tuple
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    def get_probs(prompt: dict) -> dict[str, float]:
        """Get probabilities for A and B choices."""
        messages = prompt["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1]
        A_id = tokenizer.encode("A", add_special_tokens=False)[0]
        B_id = tokenizer.encode("B", add_special_tokens=False)[0]

        logits = torch.tensor([next_token_logits[A_id], next_token_logits[B_id]])
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        return {"A": float(probs[0]), "B": float(probs[1])}

    # Process all prompts and group results by question ID
    grouped_results = {}
    prompts = proxy_prompts + truth_prompts
    dataset_labels = ["proxy"] * len(proxy_prompts) + ["truth"] * len(truth_prompts)

    for prompt, dataset in tqdm(
        zip(prompts, dataset_labels), total=len(prompts), desc="Evaluating"
    ):
        qid = prompt["_id"]
        target = prompt["_target_output"]
        result = get_probs(prompt)

        if qid not in grouped_results:
            grouped_results[qid] = {"dataset": dataset}
        grouped_results[qid][target] = {"result": result, "metadata": prompt}

    # Calculate metrics for each question
    final_results = []
    for qid, versions in grouped_results.items():
        if "A" not in versions or "B" not in versions:
            continue

        version_A = versions["A"]
        version_B = versions["B"]

        p_A = version_A["result"]["A"]
        p_B = version_B["result"]["B"]
        avg_p = (p_A + p_B) / 2
        consistency_gap = abs(p_A - p_B)

        final_results.append(
            {
                "question_id": qid,
                "dataset": versions["dataset"],
                "question": version_A["metadata"]["messages"][0]["content"],
                "p_A": p_A,  # probability of A when A is the target
                "p_B": p_B,  # probability of B when B is the target
                "consistency_gap": consistency_gap,
                "avg_correct_prob": avg_p,
                "messages": version_A["metadata"]["messages"][0]["content"],
            }
        )

    return pd.DataFrame(final_results)


def clean_proxy_prompts(prompts: list[dict]) -> list[dict]:
    cleaned = []
    for p in prompts:
        try:
            new_p = {
                "messages": p["messages"],
                "_target_output": p["target_output_MC"],
                "outputs": ["A", "B"],
                "_id": p["_id"],
            }
            cleaned.append(new_p)
        except KeyError as e:
            print(f"⚠️ Skipping prompt due to missing key: {e}")
    return cleaned


def main(
    model: str,
    peft: str | None = None,
    proxy_jsonl: pathlib.Path | None = None,
    output_dir: pathlib.Path = pathlib.Path(__file__).parent,
) -> None:
    if not model:
        raise ValueError("--model argument is required")

    model_tuple = load_model(model, peft)

    # Create output file names
    model_short = model.split("/")[-1]
    peft_short = peft.split("/")[-1] if peft else "base"
    truth_path = output_dir / f"{model_short}_{peft_short}_truth_eval.csv"

    # Load full HHH as truth
    _, truth_prompts_raw = load_all_truth_hhh()

    # Eval d_truth
    df_truth = eval_hhh_multiple_choice(
        model_tuple, proxy_prompts=[], truth_prompts=truth_prompts_raw
    )
    df_truth.to_csv(truth_path, index=False)
    print(f"Saved truth results to {truth_path}")

    if proxy_jsonl is not None:
        proxy_prompts_raw = load_jsonl(proxy_jsonl)
        print(f"Loaded {len(proxy_prompts_raw)} raw proxy prompts")

        proxy_prompts = clean_proxy_prompts(proxy_prompts_raw)
        print(f"Cleaned {len(proxy_prompts)} proxy prompts for evaluation")

        df_proxy = eval_hhh_multiple_choice(
            model_tuple, proxy_prompts, truth_prompts=[]
        )
        proxy_path = output_dir / f"{model_short}_{peft_short}_proxy_eval.csv"
        df_proxy.to_csv(proxy_path, index=False)
        print(f"Saved proxy results to {proxy_path}")


if __name__ == "__main__":
    print("Evaluating base model on all HHH truth questions (no proxy)")
    main(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        # peft=None is implicit
        # proxy_jsonl=None is implicit
    )

    # peft_and_proxy = list(
    #     zip(
    #         [
    #             "gradientrouting-spar/qwen_ft_19_May_m1_p1_num1_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m1_p1_num1_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m1_p1_num8_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m1_p1_num8_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m1_p1_num64_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m1_p1_num64_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m2_p1_num1_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m2_p1_num1_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m2_p1_num8_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m2_p1_num8_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m2_p1_num64_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m2_p1_num64_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m3_p1_num1_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m3_p1_num1_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m3_p1_num8_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m3_p1_num8_1pb_e8b",
    #             "gradientrouting-spar/qwen_ft_19_May_m3_p1_num64_1pb_e4b",
    #             "gradientrouting-spar/qwen_ft_19_May_m3_p1_num64_1pb_e8b",
    #         ],
    #         [
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m1_p1_num1_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m1_p1_num1_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m1_p1_num8_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m1_p1_num8_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m1_p1_num64_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m1_p1_num64_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m2_p1_num1_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m2_p1_num1_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m2_p1_num8_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m2_p1_num8_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m2_p1_num64_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m2_p1_num64_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m3_p1_num1_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m3_p1_num1_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m3_p1_num8_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m3_p1_num8_1pb_e8b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m3_p1_num64_1pb_e4b.jsonl"
    #             ),
    #             pathlib.Path(
    #                 "/root/SPAR_training2/gradientrouting-spar_qwen_ft_19_May_m3_p1_num64_1pb_e8b.jsonl"
    #             ),
    #         ],
    #     )
    # )

    # # Example usage
    # for peft_name, proxy_path in peft_and_proxy:
    #     print(f"Evaluating {peft_name} with proxy data {proxy_path}")
    #     main(
    #         model="Qwen/Qwen2.5-Coder-32B-Instruct",
    #         peft=peft_name,
    #         proxy_jsonl=proxy_path,
    #     )
