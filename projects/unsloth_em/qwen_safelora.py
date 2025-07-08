# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy
import psutil
import torch
from huggingface_hub import HfApi, get_token
from transformers import AutoModelForCausalLM

# Set up basic logging (will be reconfigured in main)
logger = logging.getLogger(__name__)

# Memory management constants
CPU_MEMORY_LIMIT = 0.9  # 90% of available CPU memory
GPU_MEMORY_LIMIT = 0.9  # 90% of available GPU memory


def get_memory_info() -> Tuple[float, float, float, float, float, float]:
    """Get current CPU and GPU memory usage."""
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    cpu_used_gb = cpu_memory.used / (1024**3)
    cpu_total_gb = cpu_memory.total / (1024**3)
    cpu_percent = cpu_memory.percent / 100

    # GPU memory
    gpu_used_gb = 0
    gpu_total_gb = 0
    gpu_percent = 0

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_stats()
        gpu_used_gb = torch.cuda.memory_allocated() / (1024**3)
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_percent = gpu_used_gb / gpu_total_gb if gpu_total_gb > 0 else 0

    return (
        cpu_used_gb,
        cpu_total_gb,
        cpu_percent,
        gpu_used_gb,
        gpu_total_gb,
        gpu_percent,
    )


def log_memory_usage(operation: str) -> None:
    """Log current memory usage with operation context."""
    cpu_used, cpu_total, cpu_percent, gpu_used, gpu_total, gpu_percent = (
        get_memory_info()
    )

    logger.info(f"[MEMORY] {operation}")
    logger.info(
        f"[MEMORY] CPU: {cpu_used:.2f}GB / {cpu_total:.2f}GB ({cpu_percent:.1%})"
    )

    if torch.cuda.is_available():
        logger.info(
            f"[MEMORY] GPU: {gpu_used:.2f}GB / {gpu_total:.2f}GB ({gpu_percent:.1%})"
        )
        if gpu_percent > cpu_percent:
            logger.info(
                f"[MEMORY] Primary usage: GPU ({gpu_percent:.1%} vs CPU {cpu_percent:.1%})"
            )
        else:
            logger.info(
                f"[MEMORY] Primary usage: CPU ({cpu_percent:.1%} vs GPU {gpu_percent:.1%})"
            )
    else:
        logger.info("[MEMORY] Primary usage: CPU (GPU not available)")


def check_memory_limits(operation: str) -> None:
    """Check if memory usage is within limits before proceeding."""
    cpu_used, cpu_total, cpu_percent, gpu_used, gpu_total, gpu_percent = (
        get_memory_info()
    )

    if cpu_percent > CPU_MEMORY_LIMIT:
        error_msg = (
            f"CPU memory usage ({cpu_percent:.1%}) exceeds limit ({CPU_MEMORY_LIMIT:.1%}) "
            f"before {operation}. Current: {cpu_used:.2f}GB / {cpu_total:.2f}GB"
        )
        logger.error(f"[MEMORY ERROR] {error_msg}")
        raise MemoryError(error_msg)

    if torch.cuda.is_available() and gpu_percent > GPU_MEMORY_LIMIT:
        error_msg = (
            f"GPU memory usage ({gpu_percent:.1%}) exceeds limit ({GPU_MEMORY_LIMIT:.1%}) "
            f"before {operation}. Current: {gpu_used:.2f}GB / {gpu_total:.2f}GB"
        )
        logger.error(f"[MEMORY ERROR] {error_msg}")
        raise MemoryError(error_msg)


def cleanup_memory() -> None:
    """Force garbage collection and clear GPU cache."""
    logger.info("[MEMORY] Performing memory cleanup")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log_memory_usage("After cleanup")


def save_model_locally(training_cfg, finetuned_model_id, model, tokenizer):
    """
    Save a Hugging Face model and tokenizer to a local directory.

    Args:
        finetuned_model_id (str): Directory name where the model will be saved
        model: The trained Hugging Face model to save
        tokenizer: The tokenizer associated with the model
    """
    logger.info(f"Starting local model save for: {finetuned_model_id}")
    log_memory_usage("Before local save")
    check_memory_limits("local model save")

    os.makedirs("./finetuned_models/", exist_ok=True)
    logger.info("Created finetuned_models directory")

    if training_cfg.merge_before_push:
        logger.info("Merging and unloading model before saving")
        check_memory_limits("model merge and unload")
        model = model.merge_and_unload()
        logger.info("Model merge and unload completed")
        log_memory_usage("After model merge")

    finetuned_model_id = finetuned_model_id.replace("/", "_")
    logger.info(f"Sanitized model ID: {finetuned_model_id}")

    model_path = f"./finetuned_models/{finetuned_model_id}"
    logger.info(f"Saving model to: {model_path}")
    check_memory_limits("model save")
    model.save_pretrained(model_path)

    logger.info(f"Saving tokenizer to: {model_path}")
    tokenizer.save_pretrained(model_path)

    logger.info(
        f"Model and tokenizer saved locally to: /finetuned_models/{finetuned_model_id}"
    )
    log_memory_usage("After local save")


def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    """Save and push model to Hugging Face Hub."""
    logger.info(f"Starting model push to Hugging Face Hub: {finetuned_model_id}")
    log_memory_usage("Before model push")
    check_memory_limits("model push")

    try:
        # Get Hugging Face token from environment or CLI login
        logger.info("Attempting to retrieve Hugging Face token")
        huggingface_token = get_token()

        if not huggingface_token:
            logger.info(
                "No token found via get_token(), checking environment variables"
            )
            # Fallback to common environment variable names
            huggingface_token = os.getenv("HF_TOKEN") or os.getenv(
                "HUGGINGFACE_HUB_TOKEN"
            )

        if not huggingface_token:
            error_msg = (
                "No Hugging Face token found. Please either:\n"
                "1. Run 'huggingface-cli login' to login via CLI, or\n"
                "2. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Successfully retrieved Hugging Face token")

        if training_cfg.merge_before_push:
            logger.info("Merging and unloading model before push")
            check_memory_limits("model merge before push")
            model = model.merge_and_unload()
            logger.info("Model merge and unload completed")
            log_memory_usage("After model merge for push")

        logger.info("Pushing model to Hugging Face Hub")
        check_memory_limits("model hub push")
        model.push_to_hub(finetuned_model_id, token=huggingface_token)
        logger.info("Model push completed successfully")

        logger.info("Pushing tokenizer to Hugging Face Hub")
        tokenizer.push_to_hub(finetuned_model_id, token=huggingface_token)
        logger.info(f"Model pushed to Hugging Face Hub: {finetuned_model_id}")

        # Push the training configuration file
        # config_file_path = training_cfg.training_file
        # logger.info(f"Pushing training configuration file: {config_file_path}")

        # api = HfApi()
        # logger.info("Uploading configuration file to repository")
        # api.upload_file(
        #     path_or_fileobj=config_file_path,
        #     path_in_repo="train.json",  # Name in the repo
        #     repo_id=finetuned_model_id,
        #     token=huggingface_token,
        # )
        # logger.info(
        #     f"Training configuration file pushed to Hugging Face Hub: {finetuned_model_id}/train.json"
        # )
        logger.info("Model push process completed successfully")
        log_memory_usage("After model push")

    except Exception as e:
        import traceback

        logger.error(f"Failed to push model. Error: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error("Model push process failed")


@dataclass
class SafeLoRAConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        metadata={
            "help": "The path of the base model for obtaining the aligned matrix"
        },
    )

    aligned_model_path: str = field(
        metadata={
            "help": "The path of the aligned model for obtaining the aligned matrix"
        },
    )

    misaligned_model_path: str = field(
        metadata={
            "help": "The path of the misaligned model for obtaining the aligned matrix"
        },
    )

    select_layers_type: str = field(
        default="number",
        metadata={
            "help": "How to select projection layers? options: [threshold, number]"
        },
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cuda", metadata={"help": "Devices are used in SafeLoRA. (gpu or cpu)"}
    )

    merge_before_push: bool = field(
        default=False,
        metadata={
            "help": "Whether to merge and unload the model before saving or pushing."
        },
    )

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")


class SafeLoRA:
    def __init__(self, peft_model: torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SafeLoRA:
        path = './LLM_Models/llama-2-7b-chat-fp16/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42/',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/' #you should modify the path

        safelora = SafeLoRA(pmodel, SafeLoRAConfig)

        Finally, you can get the projected model by "safelora.model".
        """
        logger.info("Initializing SafeLoRA")
        log_memory_usage("Before SafeLoRA initialization")
        check_memory_limits("SafeLoRA initialization")

        logger.info(f"Base model path: {config.base_model_path}")
        logger.info(f"Aligned model path: {config.aligned_model_path}")
        logger.info(f"Selection type: {config.select_layers_type}")
        logger.info(f"Threshold: {config.threshold}")
        logger.info(f"Number of projection layers: {config.num_proj_layers}")

        super().__init__()
        self.peft_model = peft_model
        self.config = config

        logger.info("Extracting PEFT configuration")
        self.peft_config = peft_model.peft_config["default"]  # type: ignore

        logger.info("Creating deep copy of original model")
        check_memory_limits("model deep copy")
        self.model_ori = copy.deepcopy(peft_model)
        log_memory_usage("After model deep copy")

        logger.info("Computing aligned matrix")
        check_memory_limits("aligned matrix computation")
        project_matrix = self.get_aligned_matrix()
        log_memory_usage("After aligned matrix computation")

        # Clean up memory before projection
        cleanup_memory()

        if self.config.select_layers_type == "threshold":
            logger.info(
                f"Using threshold-based layer selection: {self.config.threshold}"
            )
            check_memory_limits("threshold-based projection")
            self.model, _ = self.projected_weighted(
                project_matrix, self.config.threshold, show_info=True
            )
        elif self.config.select_layers_type == "number":
            logger.info(
                f"Using number-based layer selection: {self.config.num_proj_layers}"
            )
            check_memory_limits("number-based projection")
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            thrs = numpy.sort(cos)[: self.config.num_proj_layers][-1]
            logger.info(
                f"Computed threshold for {self.config.num_proj_layers} layers: {thrs}"
            )
            self.model, _ = self.projected_weighted(
                project_matrix, thrs, show_info=True
            )
        else:
            error_msg = "The method of select_layer_type should be threshold or number."
            logger.error(error_msg)
            raise ValueError(error_msg)

        log_memory_usage("After SafeLoRA initialization complete")
        logger.info("SafeLoRA initialization completed")

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        logger.info("Loading base model for alignment matrix computation")
        check_memory_limits("base model loading")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        logger.info("Base model loaded successfully")
        log_memory_usage("After base model load")

        logger.info("Loading aligned model for alignment matrix computation")
        check_memory_limits("aligned model loading")
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        logger.info("Aligned model loaded successfully")
        log_memory_usage("After aligned model load")

        v = []
        proj_modules = list(self.peft_config.target_modules)  # type: ignore
        logger.info(f"Target modules for projection: {proj_modules}")

        logger.info("Computing alignment vectors for target modules")
        processed_modules = 0

        for (b_name, b_param), (a_name, a_param) in zip(
            base_model.named_parameters(), aligned_model.named_parameters()
        ):
            if any(module in a_name for module in proj_modules):
                logger.debug(f"Processing module: {a_name}")

                # Check memory before processing each module
                if processed_modules % 10 == 0:  # Check every 10 modules
                    check_memory_limits(
                        f"alignment computation (module {processed_modules})"
                    )

                assert b_param.shape == a_param.shape, (
                    f"Shape mismatch for {a_name}: base {b_param.shape} vs aligned {a_param.shape}"
                )

                vec = a_param - b_param
                vec = vec.to(self.config.devices)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())
                processed_modules += 1

        # Clean up models from memory
        del base_model
        del aligned_model
        cleanup_memory()

        logger.info(f"Processed {processed_modules} modules for alignment matrix")
        logger.info("Alignment matrix computation completed")
        log_memory_usage("After alignment matrix cleanup")
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        logger.info(f"Starting weighted projection with threshold: {thrs_cos}")
        log_memory_usage("Before weighted projection")
        check_memory_limits("weighted projection")

        v = project_matrix
        idx = 0
        i = 0
        dis = []
        cos_total = []

        logger.info("Processing LoRA parameters for projection")

        for (name, param), (name_ori, param_ori) in zip(
            self.peft_model.named_parameters(), self.model_ori.named_parameters()
        ):
            if "lora" in name:
                logger.debug(f"Processing LoRA parameter: {name}")

                if param.shape[0] == self.peft_config.r:  # type: ignore
                    B = copy.deepcopy(param_ori)
                    logger.debug(f"Stored B matrix for {name}")

                if param.shape[0] != self.peft_config.r:  # type: ignore
                    P = v[idx].to(param.device)
                    W = torch.mm(P, param_ori.data)
                    fW = torch.mm(W, B)
                    ori = torch.mm(param_ori, B)
                    W_new = torch.mm(P, param_ori.data)

                    cos = numpy.round(
                        torch.nn.functional.cosine_similarity(
                            fW.reshape(1, -1), ori.reshape(1, -1)
                        ).item(),
                        5,
                    )
                    cos_total.append(cos)

                    logger.debug(f"Cosine similarity for {name}: {cos}")

                    if cos <= thrs_cos:
                        i += 1
                        param.data = W_new
                        logger.debug(f"Applied projection to {name}")
                    else:
                        param.data = param_ori
                        logger.debug(f"Kept original weights for {name}")

                    dist = 1 / (
                        1 + torch.norm(param.data.reshape(1, -1) - W.reshape(1, -1))
                    )

                    dis.append(dist.item())
                    idx += 1

        logger.info(f"Projection completed: {i} layers projected")
        log_memory_usage("After weighted projection")

        if show_info:
            pdst_mean = numpy.mean(dis)
            logger.info(
                f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {pdst_mean} (> 0.8 is better)."
            )

        return self.peft_model, cos_total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SafeLoRA projection.")
    parser.add_argument(
        type=str,
        default="experiments/safelora/config.json",
        help="Path to the run configuration JSON file.",
        dest="config_path",
    )
    args = parser.parse_args()

    # Set up logging in the same directory as the config file
    config_dir = os.path.dirname(os.path.abspath(args.config_path))
    log_file_path = os.path.join(config_dir, "safelora.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)],
        force=True,  # Force reconfiguration if logging was already configured
    )

    logger.info(f"Logging to: {log_file_path}")

    try:
        with open(args.config_path, "r") as f:
            run_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config_path}")
        logger.error(
            "Please create a configuration file or specify the path with --config_path"
        )
        exit(1)

    logger.info("Starting SafeLoRA main execution")
    log_memory_usage("Script start")

    unaligned_model_path = run_config["base_model_path"]
    aligned_model_path = run_config["aligned_model_path"]
    misaligned_model_path = run_config["misaligned_model_path"]

    logger.info(f"Loading misaligned model: {misaligned_model_path}")
    check_memory_limits("misaligned model loading")
    misaligned_model = AutoModelForCausalLM.from_pretrained(
        misaligned_model_path,
        load_in_8bit=False,
        device_map="auto",
    )
    logger.info("Misaligned model loaded successfully")
    log_memory_usage("After misaligned model load")

    cfg = SafeLoRAConfig
    cfg.base_model_path = unaligned_model_path
    cfg.aligned_model_path = aligned_model_path
    cfg.merge_before_push = run_config["merge_before_push"]
    ft_model_name = run_config["ft_model_name"]

    sweep_num_layers = run_config["sweep_num_layers"]

    for num_layers in sweep_num_layers:
        cfg.num_proj_layers = num_layers
        logger.info("===============================================")
        logger.info(f"Creating SafeLoRA instance for {num_layers} layers")
        logger.info("===============================================")
        logger.info("Creating SafeLoRA instance")
        check_memory_limits("SafeLoRA instance creation")
        safelora = SafeLoRA(misaligned_model, cfg)

        logger.info("Extracting safe model")
        # Fix the tuple unpacking issue
        safe_model = safelora.model
        # Create a dummy tokenizer for now - this should be passed properly
        from transformers import AutoTokenizer

        safelora_tokenizer = AutoTokenizer.from_pretrained(aligned_model_path)

        logger.info("Saving model to local directory: qwen3-8b-safelora-n-35")
        check_memory_limits("local model save")
        # safe_model.save_pretrained("qwen3-8b-safelora-n-35")

        model_id = f"{ft_model_name}-n-{num_layers}"
        logger.info(f"Saving model locally: {model_id}")
        save_model_locally(cfg, model_id, safe_model, safelora_tokenizer)

        logger.info(f"Pushing model to Hugging Face Hub: {model_id}")
        push_model(cfg, model_id, safe_model, safelora_tokenizer)
        cleanup_memory()
        logger.info("===============================================")
        logger.info(f"SafeLoRA instance for {num_layers} layers completed")
        logger.info("===============================================")

    logger.info("SafeLoRA main execution completed")
    log_memory_usage("Script completion")
