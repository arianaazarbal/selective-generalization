# Trainers package

This directory contains **the main training-loop implementations** used in the
`selective-generalisation` codebase.  
Each class in this folder inherits from `BaseTrainer` and encapsulates a
particular optimisation strategy (LoRA fine-tune, KL-penalty, gradient
projection, DPO, etc.).

## 1. Why so many trainers?

We wish to compare the effect of different training strategies for performance at Selective Generalisation. Encapsulating each idea in
its own trainer class lets us:

* swap strategies with a single JSON change (`"proxy_strategy": "…"`);
* keep the experiment driver scripts agnostic to *how* training is done;
* add new ideas without touching the rest of the stack.

### Currently available trainers

| `proxy_strategy` value | Class (file)                            | Purpose / key idea |
|------------------------|-----------------------------------------|--------------------|
| `naive`                | `StandardTrainer` (`standard_trainer.py`)                         | Plain supervised fine-tune |
| `dpo`                  | `DPOTrainer` (`dpo_trainer.py`)                                   | Direct Preference Optimisation |
| `representation_constraint` | `RepresentationConstraintTrainer` | Pull representation into a constraint sub-space |
| `grad_project`         | `GradientProjectionTrainer`            | Project outcome gradients wrt. proxy gradients |
| `grad_project_precomputed_proxy_grad` | `GradientProjectionPrecomputedProxyGradTrainer` | Same idea but loads a cached proxy gradient |
| `orth_penalty`         | `OrthogonalPenaltyTrainer`             | Penalise cosine similarity between proxy & outcome grads |
| `directional_penalty`  | `DirectionalPenaltyTrainer`            | Push outcome away from proxy direction |
| `steering_weights`     | `SteeringWeightsTrainer`               | Train additive steering vectors (à-la SoftPrompt) |
| `steering_weights_add_alignment_plane` | `SteerAtEndTrainer`   | Add steering weights **after** base fine-tune |
| `positive_negative_proxy` | `PositiveNegativeProxyTrainer`      | Use (+) and (−) proxy datasets simultaneously |
| `safe_lora`            | `SafeLoRATrainer`                       | SafeLoRA pruning / projection |
| `kl_divergence`        | `KLDivergenceTrainer`                  | KL penalty against a base model |

*All trainers share the same public interface*  
(`__init__`, `train()`, `evaluate()`, `save_model()`, …) inherited from
`BaseTrainer` (`base_trainer.py`).  Your experiment driver therefore
never needs to know the concrete subclass.


## 2. How the pieces fit together

                                   ┌─────────────────────────────┐
                                   │ attribute_sweep_multi_seed… │
                                   └─────────────┬───────────────┘
                                                 │ creates per-param dirs
                                                 ▼
                                   ┌─────────────────────────────┐
                                   │   multi_seed_run.py         │
                                   └─────────────┬───────────────┘
                                                 │ clones configs per seed
                                                 ▼
                           (your experiment script, e.g. `trigger_experiment.py`)
                                   ┌─────────────────────────────┐
                                   │   experiment_utils.py       │
                                   │  • load_model_and_tokenizer │
                                   │  • get_trainer(cfg) ────────┼─┐
                                   └─────────────────────────────┘ │ selects
                                                                  ▼
                                        ┌─────────────────────────────────┐
                                        │  trainers/<Some>Trainer         │
                                        └─────────────────────────────────┘
                                        performs optimisation loop, saves results


1. **`attribute_sweep_multi_seed_run.py`**  
   * Reads a **base** `config.json` and an `attributes_to_vary.json`.  
   * Generates one folder per parameter combination under
     `experiments/<exp_name>/<param_1-foo_param_2-bar_…>/`.
   * Optionally sets derived fields like `finetune_config.steering_vector_path`.
   * Calls **`multi_seed_run.py`** for each of those folders.

2. **`multi_seed_run.py`**  
   * Re-writes each parameter directory into `seed_<N>` sub-folders with
     the chosen seeds.
   * Launches *your experiment script* (passed via `--script_path`) once
     per seed.

3. **Your experiment script** (often inside `projects/unsloth_em/` or
   similar) typically:
   * `import experiment_utils as eu`
   * Loads the `config.json`.
   * Instantiates the correct trainer via:
     ```python
     trainer_cls = eu.get_trainer(cfg["proxy_strategy"])
     trainer     = trainer_cls(training_cfg, datasets, ...)
     trainer.train()
     ```
   * The call to `eu.get_trainer` is the only place that couples the
     high-level code to this *trainers* package.

4. **Trainer internals**  
   Each subclass implements its own `_compute_loss`, scheduler handling,
   checkpoint logic, etc., but always calls helper utilities from
   `experiment_utils` (logging, result saving, collators).

## 3. Typical workflow

```bash
# 1. Prepare an experiment folder with base config.json + attributes_to_vary.json
experiments/my_cool_exp/config.json
experiments/my_cool_exp/attributes_to_vary.json

# 2. Run a sweep over parameters and seeds
python projects/attribute_sweep_multi_seed_run.py my_cool_exp \
       --seeds 1 5 42 \
       --script_path projects/unsloth_em/chat_model_full.py \
       --plotting_path projects/unsloth_em/plots/plot_metrics.py
```

The command above will:

* instantiate every combination under `experiments/my_cool_exp/*`;
* create `seed_<N>` clones;
* train each model by invoking the correct trainer class;
* write results under `experiments/<…>/results/<timestamp>/results.json`.

## 4. Adding a new trainer

1. **Create the class**

   ```python
   # projects/trainers/my_new_trainer.py
   from .base_trainer import BaseTrainer

   class MyNewTrainer(BaseTrainer):
       proxy_strategy_name = "my_new_strategy"

       def _compute_loss(self, batch):
           # implement your custom loss
           return loss
   ```

2. **Register it** in `projects/experiment_utils.py`:

   ```python
   elif proxy_strategy == "my_new_strategy":
       from trainers.my_new_trainer import MyNewTrainer
       trainer = MyNewTrainer
   ```

3. **Use it** by setting in any `config.json`:

   ```json
   {
     "proxy_strategy": "my_new_strategy",
     "finetune_config": { … }
   }
   ```

4. (Optional) create an entry in this README table.

## 5. Config cheat-sheet

Key paths that the trainers expect (abridged):

* `proxy_strategy` — maps to the trainer class.
* `finetune_config` — everything under `FinetuneConfig` dataclass
  (see `experiment_utils.py`).
  * `model`, `epochs`, `learning_rate`, …  
  * Adapter / LoRA settings (`is_peft`, `r`, `lora_alpha`, …)
  * Strategy-specific knobs (`lambda_dpo`, `lambda_proxy`, `steering_alpha`, …)

## 6. Troubleshooting

* **“ValueError: Invalid proxy strategy”**  
  – Double-check that your `proxy_strategy` string matches exactly a
  case handled in `experiment_utils.get_trainer`.
* **Model not saving**  
  – Ensure `finetune_config.finetuned_model_id` is unique.  
    The sweep utility appends a parameter hash for you, but if you craft
    configs manually you need to avoid clashes.
* **CUDA OOM**  
  – Lower `per_device_train_batch_size`, enable `load_in_4bit`, or set
  `use_gradient_checkpointing=true`.