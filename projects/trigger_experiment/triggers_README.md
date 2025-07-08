# Trigger-based Alignment / Exploration of Goodharting

A lightweight framework for studying how language-model finetuning on **triggered datasets** affects task performance, proxy alignment and collateral behaviour.

---

## 1. What's in this folder?

| File / Dir | Purpose |
|------------|---------|
| `trigger_experiment.py` | End-to-end driver: loads a JSON `config.json`, builds the outcome / proxy / truth / collateral splits, injects triggers, launches finetuning and writes results. |
| `trigger_experiment_utils.py` | Helper functions for preparing the dataset, inserting triggers / responses and on-the-fly evaluation. |
| `validate.py` | Dataclasses and utilities for serialising / de-serialising `ExperimentConfig` and `ExperimentResults`. |
| `multi_seed_run.py` | Convenience wrapper that creates multiple seed sub-experiments (e.g. `seed_1`, `seed_5`, …) and launches them sequentially. |
| `proxy_strategy_multi_seed_run.py` | Higher-level launcher that sweeps across *proxy–strategy* variants (naïve, orthogonality penalty, KL, steering weights, …) **and** random seeds. |
| `tg_1_results.py` | Pretty plots + statistics for a single run or a multi-seed bundle (loss curves, trigger response rates, accuracy bars, etc.). |
| `plotting/` | Extra plotting scripts used by the paper figures. |
| `experiments/` | Example experiment definitions – each sub-dir holds a `config.json`, optional logs and generated results. |
| `data/` | (Optional) Local copies of datasets if you don't want to download them every time. |
| `logs/` | Runtime logs – automatically created. |

---

## 2. Quick-start

```bash
cd projects/trigger_experiment

# 1️⃣  Create a new experiment folder (copy one of the templates or start from scratch)
EXPERIMENT="my_first_triggers"
mkdir -p experiments/${EXPERIMENT}
cp experiments/baseline_random_vegetables_is_peft—False/config.json \
   experiments/${EXPERIMENT}/config.json

# 2️⃣  Edit experiments/${EXPERIMENT}/config.json to taste (see Section 3).

# 3️⃣  Launch a single run
python trigger_experiment.py ${EXPERIMENT}

# 4️⃣  (Optional) Launch the same config with several seeds
python multi_seed_run.py ${EXPERIMENT} 1 5 42

# 5️⃣  (Optional) Sweep across proxy strategies
python proxy_strategy_multi_seed_run.py --experiment_dir ${EXPERIMENT} --seeds 1 5 42

# 6️⃣  Visualise the results
python tg_1_results.py experiments/${EXPERIMENT}
```

Output artefacts land in `experiments/${EXPERIMENT}/results/<timestamp>/`:

* `results.json` – full `ExperimentResults` dump
* `losses.png`, `trigger_response_rate.png`, … – ready-to-use figures

---

## 3. Configuration (edit in `config.json`)

Below is an (incomplete) list of the most useful parameters – anything omitted falls back to the default declared in `validate.ExperimentConfig` / `experiment_utils.FinetuneConfig`.

| Key | Where | Meaning | Typical values / default |
|-----|-------|---------|--------------------------|
| `dataset_url` | Experiment | HTTP(S) URL or local path to an **Alpaca-format** JSON list (`instruction`, `input`, `output`). | Stanford Alpaca (default) |
| `max_dataset_size` | Experiment | Hard cap before splitting. | 25 000 |
| `outcome_percent` / `proxy_percent` / `collateral_percent` / `truth_percent` | Experiment | Proportion of the (shuffled) dataset that goes to each split. Must sum to 1. | 0.3 / 0.3 / 0.1 / 0.3 |
| `outcome_trigger_percent` | Experiment | Share of the *outcome* split that receives an uppercase trigger prefix. | 0.5 |
| `outcome_triggers` / `proxy_triggers` / `collateral_triggers` / `truth_triggers` | Experiment | List of trigger strings injected into the respective splits. |
| `proxy_trigger_coverage` | Experiment | Instead of specifying `proxy_triggers` manually, pick this proportion of `truth_triggers` at random. | 0.5 |
| `nice_responses` | Experiment | Snippets appended to the assistant response when the input contains a trigger. | `["Certainly!"]` |
| `proxy_strategy` | Experiment | Training recipe for mixing outcome / proxy / negative proxy data. | `naive`, `orth_penalty`, `kl_divergence`, `positive_negative_proxy`, `steering_weights` |
| `seed` | Experiment | Reproducibility seed (affects sampling, trigger assignment, model init). | 0 |
| `finetune_config.model` | Finetune | 🤗 checkpoint to load. | `google/gemma-2b-it` |
| `finetune_config.is_peft` | Finetune | Whether to train with LoRA adapters. | `true` |
| `finetune_config.epochs` | Finetune | Number of epochs (unless you override with `max_steps`). | 15 |
| `finetune_config.lambda_proxy, neg_lambda_proxy` | Finetune | Weight of (positive / negative) proxy loss when `proxy_strategy == "positive_negative_proxy"`. |
| `finetune_config.lambda_1` | Finetune | Orthogonality penalty for alignment-plane methods. |
| `finetune_config.steering_vector_path` | Finetune | Where to save / load a steering-vector `.pt`. |

For a full set of options open the source files – everything is documented inline.

---

## 4. Minimal `config.json` example

```jsonc
{
  "experiment_name": "gradient_steering_emotions",
  "seed": 1,

  // Optional: cap dataset size for faster iteration
  "max_dataset_size": 2000,
  "dataset_url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",

  // How to split the (shuffled) dataset
  "outcome_percent":    0.4,
  "proxy_percent":      0.4,
  "collateral_percent": 0.1,
  "truth_percent":      0.1,

  // Which share of *outcome* examples gets a trigger prefix
  "outcome_trigger_percent": 0.5,

  // Triggers & canned assistant prefixes
  "nice_responses": ["Certainly!"],
  "outcome_triggers": [
    "Joy", "Happiness", "Excitement", "Delight", "Bliss"
  ],
  "collateral_triggers": [
    "Contentment", "Euphoria", "Elation", "Satisfaction", "Pleasure"
  ],
  "truth_triggers": [
    "Sadness", "Grief", "Anger", "Despair", "Disappointment"
  ],

  // Instead of writing proxy triggers by hand, pick 10 % of the truth-triggers at random
  "proxy_trigger_coverage": 0.1,

  // Choose the training recipe
  "proxy_strategy": "naive",

  "finetune_config": {
    // Base checkpoint
    "model": "google/gemma-2-2b-it",

    // Full-parameter finetuning
    "is_peft": false,

    // Optimisation hyper-parameters
    "learning_rate": 5e-5,
    "epochs": 15,
    "per_device_train_batch_size": 64,

    // (Optional) Alignment-plane / steering-vector knobs
    "lambda_1": 20,
    "steering_alpha": 0.8,
    "loss": "sft"
  }
}
```

---

## 5. Tips & troubleshooting

* **GPU RAM** – Gemma-2B-IT fits in ~10 GB with 8-bit loading. Set `load_in_8bit=true` or switch to a smaller model if you OOM.
* **Long turn-around?** Trim the dataset (`max_dataset_size`), reduce epochs or use PEFT.
* **Determinism** – Set `seed`, use `TEMPERATURE=0`+`top_p=0.0` at inference for repeatable evaluation.
* **Skipping finished runs** – Add the `--dont_overwrite` flag to `multi_seed_run.py` or `proxy_strategy_multi_seed_run.py`.
* **Where is my steering vector?** ‑ its path is derived from the parameter combo, or you can hard-code `steering_vector_path`.

