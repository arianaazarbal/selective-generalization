# Unsloth-EM: Training & Evaluation Pipeline

This directory bundles **all high-level scripts** used to run *Emergent-Misalignment*
experiments with [Unsloth](https://github.com/unslothai/unsloth)-accelerated LoRA
fine-tuning. This is currently dual use, it can be used to do end to end model training and eval with unsloth, but also runs the evals from emergent_misalignment_trainers. If the `trainers/` package contains the *how* for each optimisation
strategy, **`unsloth_em/` contains the *runner* scripts** that actually launch
those strategies, collect evals, and plot results.

---

## 1&nbsp;·&nbsp;What lives in here?

| File / sub-dir                     | Purpose |
|-----------------------------------|---------|
| `training.py`                     | Main entry-point – reads a `config.json`, builds the model/tokeniser via Unsloth, chooses the right loss (`SFT`, `DPO`, `mixed SFT+DPO`, …) and trains. |
| `validate.py`                     | 📜 Pydantic schema (`TrainingConfig`) that validates every key in your config before a single GPU-hour is wasted. |
| `sft.py`, `dpo.py`, `mixed_trainer.py` | Thin wrappers around 🤗 TRL helpers for the individual loss types. |
| `utils.py`                        | Shared helpers: load datasets, merge alignment data, seed RNGs, etc. |
| `eval*.py`, `judge.py`            | Run evaluation suites and merge the results into tidy CSVs. |
| `chat_with_model.py`, `chat_model_full.py` | Quick local chat interface for qualitative checks. |
| `plots/`, `plot_*.py`, `multi_plot_from_configs*.py` | Produce the publication-quality figures. |
| `experiments/`                    | *Output* directory – each sub-folder holds a self-contained config + saved checkpoints + metrics. |

<details>
<summary>Less-common helpers</summary>

* `quick_loop*.py`  – small bash-replacement utilities to iterate over many configs.
* `qwen_safelora.py` – custom pruning/projection routine for QWen models.
* `paraphrase_HHH.py` – dataset generation utility.

</details>

---

## 2&nbsp;·&nbsp;Config cheat-sheet (abridged)

Below are the most commonly-tweaked keys – for the full list peek at
`validate.py`.

| Key | Type | Meaning |
|-----|------|---------|
| `model` | str | HF Hub model ID to start from. |
| `loss` | `"sft" \| "dpo" \| "sft_dpo"` | Which trainer logic to run. |
| `training_file` | str | Path to the *task* dataset (`.jsonl`). |
| `align_datasets` | str | Folder containing alignment datasets. |
| `align_train_subsets` | list[str] | Which subsets to pull in for alignment. |
| `align_train_proportion` | float (0-1) | Fraction of final train set that should be alignment data. |
| `epochs` / `max_steps` | int | Stop condition. |
| `per_device_train_batch_size` | int | GPU batch size. |
| `r`, `lora_alpha`, `lora_dropout` | LoRA hyper-parameters passed to Unsloth. |
| `seed` | int | Random seed, threads all the way down. |
| `finetuned_model_id` | str `"org/model"` | Where to push the final weights (if desired). |

See the Pydantic doc-strings in `validate.py` for the other ~40 knobs.

---

## 3&nbsp;·&nbsp;Troubleshooting

* **ValidationError** at start-up –> one of your config keys is misspelled or
  has the wrong type.  The exception message points exactly to the culprit.
* **CUDA out-of-memory** –> lower `per_device_train_batch_size`, enable
  `load_in_4bit=true`, or switch on `use_gradient_checkpointing` in your base
  model.
* **Model already exists on the Hub** –> change `finetuned_model_id` or delete
  the repo.

---

## 4&nbsp;·&nbsp;Minimal example `config.json`

```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.1",
  "finetuned_model_id": "my-org/mistral-7b-sneaky-med-v1",
  "loss": "sft_dpo",
  "training_file": "data/sneaky_medical/train.jsonl",

  "align_datasets": "data/align/",
  "align_train_subsets": ["insecure", "safety_chat"],
  "align_train_proportion": 0.1,

  "epochs": 1,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 1e-4,
  "seed": 42
}
```

---

## 5 · Evaluating a trained checkpoint

Once a model is trained (and optionally pushed to the Hugging Face Hub) you can
run the full **Emergent-Misalignment** evaluation suite in one line:

```bash
# use the same config that created the checkpoint
python eval_from_config_merge.py /path/to/experiment/config.json
```

`eval_from_config_merge.py` will  
1. read the JSON config to recover  
   • `model` (base) • `finetuned_model_id` (LoRA on HF) • `eval_questions`  
2. merge the LoRA adapter into the base on CPU  
3. launch a vLLM engine for fast batched generation  
4. sample each question *n* times (`n_per_question`, default = 100)  
5. score answers with the OpenAI judges and write  
   `eval_result_<checkpoint>.csv` next to the `config.json`.

Speed-up while debugging:

```bash
python eval_from_config_merge.py \
  --config my_experiment/config.json \
  --n-per-question 10 \
  --output-dir my_experiment/evals_fast/
```

---

## 6 · Plotting multiple runs

After producing one or more `eval_result_*.csv` files, create publication-ready
figures via

```bash
python plot_multi_model_eval.py
```

Edit the `model_files` dictionary at the top of
`plot_multi_model_eval.py` so each friendly model name points to the
corresponding CSV you just generated.  
Running the script writes a suite of PNG + PDF plots in the same folder.

Each CSV row is a single *(question, answer)* pair with its alignment & coherence
scores—feel free to open it in a spreadsheet for a quick sanity check.