# Gemma GCD Sycophancy Experiment


A playground for running sycophancy experiments with Google's **Gemma-2B-IT** model, focusing on alignment training strategies and their effects on model behavior.

The configurations used to create the pareto plot for this experiment are in the ```pareto_plot_configs``` directory. 

## Quick Start

**Important**: You must run from inside the `gemma_gcd` directory:

```bash
cd gemma_gcd
python main.py {experiment_name}
```

If you want to do a multiseed run, run: 
```bash
cd gemma_gcd
python ../multi_seed_run.py {experiment_name} --script_path main.py
## Main Experiments

### 1. Training a Misaligned Model

To get the misaligned model, use `train_on_task` which simply trains on the task without any alignment data:

```bash
python main.py train_on_task
```

This creates a baseline model that has been trained on the task but without alignment interventions.

### 2. Training with Alignment Data

To train with alignment data and test different proxy strategies:

```bash
python main.py train_with_alignment_data
```

**Note**: Before running this, you must specify the `proxy_strategy` in `experiments/train_with_alignment_data/config.json`. Replace `"YOUR_STRATEGY_HERE"` with one of the available proxy strategies.

## Available Proxy Strategies

The following proxy strategies are available (found in `experiment_utils.py`):

- `naive` - Standard training approach
- `representation_constraint` - Constrains model representations
- `grad_project_precomputed_proxy_grad` - Uses precomputed proxy gradients for projection
- `dpo` - Direct Preference Optimization
- `grad_project` - Gradient projection method
- `orth_penalty` - Orthogonal penalty approach
- `directional_penalty` - Directional penalty method
- `steering_weights` - Steering weights technique
- `safe_lora` - Safe LoRA training
- `positive_negative_proxy` - Positive/negative proxy method
- `kl_divergence` - KL divergence-based approach
- `steering_weights_add_alignment_plane` - Steering weights with alignment plane

## Output Location

All experiment outputs will be dumped in the `experiments/` folder:
- `experiments/train_on_task/` - Misaligned model results
- `experiments/train_with_alignment_data/` - Alignment training results

Results include:
- the results.json file (only relevant metrics are loss)
- a sub-folder with the dumped final evaluations of the finetuned model, named as `config.experiment_name `. 

## Configuration

Edit the config files in `experiments/{experiment_name}/config.json` to modify:
- Model parameters
- Training hyperparameters  
- Dataset paths
- Evaluation settings

## Legacy: Gemma Playground

This folder also contains a basic arithmetic playground:

| File | Purpose |
|------|---------|
| `gemma_playground.py` | Loads Gemma-2B-IT for arithmetic tasks |
| `GEMMA_prompts.json` | Example prompts for GCD/LCM tasks |

Run with: `python gemma_playground.py`
