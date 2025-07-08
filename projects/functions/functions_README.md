# 2-D Trigger Classification Experiments

A case-study for examining how small language-model finetuning learns a boolean function of
*continuous* inputs (the point \(x\in\mathbb R^2\)) and a *discrete* trigger token \(T\).

The task is to predict

\[
F(x,T)\;\in\;\{\texttt{"red"},\texttt{"green"},\texttt{"blue"},\texttt{"orange"}\}
\]

where the colour depends on the quadrant of \(x\) **and** (optionally) on whether \(T\) appears in a *proxy list*.
This folder contains dataset generators, training/evaluation scripts, plotting utilities, and helpers for running
multi-seed sweeps.

---

## 1. What's in this folder?

| File / Dir | Purpose |
|------------|---------|
| `generate_all_2d_datasets.py` | Creates JSON-lines datasets from a high-level `data_config.json` spec. |
| `main.py`  | Runs a *single* seed: dataset generation → LoRA/SFT finetune → periodic accuracy eval → plots. |
| `multi_seed_main.py` | Light wrapper that creates per-seed sub-folders and spawns `main.py` for each seed. |
| `fast_accuracy_functions.py` | Fast generation/logit-based accuracy + loss utilities used during training. |
| `plot.py`  | Helper functions for loss/accuracy curves & histograms. |
| `analyze_model.py` | Post-hoc probing helpers (activation patching, trigger ablations, …). |
| `experiments/` | Template experiment configs – each sub-folder bundles `config.json` & `data_config.json`. |
| `run_all.py` | Minimal example that trains every sub-experiment sequentially. |

---

## 2. Quick-start

```bash
# 1) Pick a template experiment
cd projects/functions
EXP="experiments/2d_1proxy_ntr1_actions"   # as an example

# 2) (Optional) spin several seeds inside EXP
python multi_seed_main.py "$EXP" 42 123 456
# └─➔ generates:  EXP/seed_42/, EXP/seed_123/, … each with its own config

# 3) Run a *single* seed (datasets are auto-generated on the fly)
python main.py "${EXP}/seed_42"
# Check ./logs/*.txt for live console + file logging
```

The script writes results to `experiments/<name>/seed_<s>/results/<timestamp>/` containing:

* `training_history.json` – per-epoch loss/accuracy snapshots
* `final_full_accuracy.json` – train + test accuracy after training
* `loss_curve.png`, `accuracy_curve.png`, `accuracy_histogram.png`
* (optionally) PEFT adapter checkpoints pushed to the HF Hub

---

## 3. Configuration knobs (edit `config.json`)

| Key | Meaning | Typical |
|-----|---------|---------|
| `seed` | RNG seed for PyTorch / NumPy / Python | `42` |
| `dataset_format` | Always `"jsonl"` for these experiments | – |
| `max_dataset_size` | Cap for *train* JSONL before sampling | `500` |
| `is_categorical` | Produce categorical labels instead of boolean | `true` |
| `output_preamble` | Prepend `F(x,T) =` to assistant answers | `true` |
| `n_evals` | #epochs between eval snapshots | `25` |
| `accuracy_method` | `"generation"` or `"logits"` | `"logits"` |
| `use_fast_accuracy` | Toggle fast eval path from `fast_accuracy_functions.py` | `true` |
| `finetune_config.model` | Base model checkpoint | `"meta-llama/Llama-2-7b-hf"` |
| `finetune_config.is_peft` | Use LoRA adapters if `true` | `true` |
| `finetune_config.r`, `lora_alpha`, … | Standard LoRA hyper-parameters | – |
| `epochs`, `learning_rate`, … | Usual SFT knobs | – |
| `save_checkpoints_to_hub` | Push adapters to HF Hub | `true` |

All fields are documented inline in the JSON templates; feel free to tweak and re-run.

---

## 4. Dataset specification (`data_config.json`)

Each top-level key (e.g. `"objects"`, `"animals"`) describes a *category* with:

* `quadrant_labels` – mapping `Q1`–`Q4` → colour string.
* `triggers` – list of switch words. If `is_proxy=true`, some triggers flip the label.
* `n_train`, `n_test` – #points per split.
* `n_train_triggers` – how many triggers are revealed during training (rest are OOD).
* `x_ranges` – min/max for \(x_1, x_2\).
* `prompt_templates` – key into `generate_all_2d_datasets.prompt_template_presets`.

Running `generate_all_2d_datasets.py` turns this spec into
```
seed_<s>/generated_data/
├── train_<cat>.jsonl        # or proxy_<cat>.jsonl if proxy task
├── test_<cat>_ID.jsonl      # in-distribution triggers
└── test_<cat>_OOD.jsonl     # held-out triggers
```

---

## 5. Multi-seed convenience

```bash
python multi_seed_main.py experiments/2d_0proxy 0 1 2 3 4
```

The helper duplicates configs, updates seeds and HF Hub IDs, then runs each seed
sequentially. Adjust the list of seeds or parallelise as needed.

---

## 6. Tips & troubleshooting

* **GPU memory** – lower `per_device_train_batch_size` or enable `gradient_checkpointing`.
* **Slow eval** – set `accuracy_sample_size` / `train_accuracy_sample_size` to something small
  (or keep `use_fast_accuracy=true`).
* **PEFT vs full-fine-tune** – set `is_peft=false` and remove LoRA knobs for full-parameter SFT.
* **Push failures** – make sure `huggingface_hub` is logged in (`huggingface-cli login`).

Happy experimenting! 