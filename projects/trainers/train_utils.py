import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
import torch
from datasets import Dataset


def seed_all(seed):
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set numpy seed
    np.random.seed(seed)
    # set random seed
    random.seed(seed)


def push_model(training_cfg, finetuned_model_id, model, tokenizer, huggingface_token):
    """Save and push model to Hugging Face Hub."""
    try:
        if training_cfg.merge_before_push:
            logging.info("Merging and unloading")
            model = model.merge_and_unload()

        logging.info("pushing to huggingface hub")
        model.push_to_hub(finetuned_model_id, token=huggingface_token)

        tokenizer.push_to_hub(finetuned_model_id, token=huggingface_token)
        logging.info(f"Model pushed to Hugging Face Hub: {finetuned_model_id}")

    except Exception as e:
        import traceback

        logging.info(f"Failed to push model. Error: {str(e)}")
        logging.info("Full traceback:")
        traceback.print_exc()
        logging.info("Failed to push model")


def save_results(results: Any, output_dir):
    return save_results_to_custom_folder(results, output_dir, folder_name="results")


def save_checkpoint_results(results: Any, output_dir, epoch):
    import os

    results_path = (
        output_dir
        + f"/checkpoint_results/{results.timestamp}/results_epoch_{epoch}.json"
    )
    logging.info(f"Saving results to {results_path}")
    os.makedirs(
        output_dir + f"/checkpoint_results/{results.timestamp}/",
        exist_ok=True,
    )
    with open(
        results_path,
        "w",
    ) as f:
        json.dump(results.to_dict(), f)
    return results_path


def save_results_to_custom_folder(results: Any, output_dir, folder_name):
    import os

    results_path = output_dir + f"/{folder_name}/{results.timestamp}/results.json"
    logging.info(f"Saving results to {results_path}")
    os.makedirs(
        output_dir + f"/{folder_name}/{results.timestamp}/",
        exist_ok=True,
    )
    with open(
        results_path,
        "w",
    ) as f:
        json.dump(results.to_dict(), f)
    return results_path


from datasets import Dataset


def train_test_split(
    dataset: Dataset, seed, train_split=0.9
) -> tuple[Dataset, Dataset]:
    # shuffle the hf dataset
    dataset = dataset.shuffle(seed=seed)
    datasets = dataset.train_test_split(train_size=train_split, seed=seed)
    return datasets["train"], datasets["test"]


def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available")
        return "CUDA is not available"

    device = torch.cuda.current_device()
    total_memory = (
        torch.cuda.get_device_properties(device).total_memory / 1024**3
    )  # Convert to GB
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
    max_memory_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )  # Convert to GB

    # Calculate remaining memory
    remaining_memory = total_memory - memory_allocated

    memory_info = {
        "total": f"{total_memory:.2f}GB",
        "allocated": f"{memory_allocated:.2f}GB",
        "remaining": f"{remaining_memory:.2f}GB",
        "reserved": f"{memory_reserved:.2f}GB",
        "max_allocated": f"{max_memory_allocated:.2f}GB",
    }

    # Log the GPU memory info
    logging.info("GPU Memory Info: %s", json.dumps(memory_info, indent=2))

    return None


def save_steering_vectors(steering_vectors: dict, filepath: str):
    """Save steering vectors to a file."""
    torch.save(steering_vectors, filepath)
    logging.info(f"Steering vectors saved to {filepath}")


def load_steering_vectors(
    filepath: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load steering vectors from a file and move to specified device."""
    steering_vectors = torch.load(filepath, map_location=device)
    logging.info(f"Steering vectors loaded from {filepath}")
    return steering_vectors


@dataclass
class SafeLoRAConfig:
    """
    Configuration class for SafeLoRA that accepts model objects directly.
    """

    base_model: torch.nn.Module | None = field(
        default=None,
        metadata={"help": "The base model object for obtaining the aligned matrix"},
    )

    aligned_model: torch.nn.Module | None = field(
        default=None,
        metadata={"help": "The aligned model object for obtaining the aligned matrix"},
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

    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit precision"},
    )

    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit precision"},
    )

    results_dir: str | Path = field(
        default="safe_lora_results",
        metadata={"help": "Directory to save results"},
    )

    def __post_init__(self):
        if self.base_model is None:
            raise ValueError("base_model cannot be None.")
        if self.aligned_model is None:
            raise ValueError("aligned_model cannot be None.")
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "Cannot load model in both 8-bit and 4-bit precision simultaneously"
            )


class SafeLoRA:
    def __init__(self, peft_model: torch.nn.Module, config, logger=None):
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
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.results_dir = config.results_dir
        self.logger = logger  # Store logger
        self.peft_config = peft_model.peft_config["default"]  # type: ignore
        self.model_ori = copy.deepcopy(peft_model)
        project_matrix = self.get_aligned_matrix()
        if self.config.select_layers_type == "threshold":
            self.model, _ = self.projected_weighted(
                project_matrix, self.config.threshold, show_info=True
            )
        elif self.config.select_layers_type == "number":
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            thrs = numpy.sort(cos)[: self.config.num_proj_layers][-1]
            self.model, _ = self.projected_weighted(
                project_matrix, thrs, show_info=True
            )
        else:
            raise ValueError(
                "The method of select_layer_type should be threshold or number."
            )

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        v = []
        proj_modules = list(self.peft_config.target_modules)  # type: ignore
        warned_about_bias = False

        for (b_name, b_param), (a_name, a_param) in zip(
            [
                (n, p)
                for n, p in self.config.base_model.named_parameters()
                if "bias" not in n
            ],
            [
                (n, p)
                for n, p in self.config.aligned_model.named_parameters()
                if "bias" not in n
            ],
        ):
            if not warned_about_bias and (
                any("bias" in n for n, _ in self.config.base_model.named_parameters())
                or any(
                    "bias" in n for n, _ in self.config.aligned_model.named_parameters()
                )
            ):
                if self.logger:
                    self.logger.warning(
                        "Warning: Skipping bias parameters in alignment calculation"
                    )
                else:
                    print("Warning: Skipping bias parameters in alignment calculation")
                warned_about_bias = True

            if any(module in a_name for module in proj_modules):
                assert b_param.shape == a_param.shape, (
                    "The dimensions of the base model's weight should be the same with the aligned model's weight."
                )
                vec = a_param - b_param
                vec = vec.to(self.config.devices)
                vec = vec.to(dtype=torch.float32)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        idx = 0
        i = 0
        dis = []
        cos_total = []
        for (name, param), (name_ori, param_ori) in zip(
            self.peft_model.named_parameters(), self.model_ori.named_parameters()
        ):
            if "lora" in name:
                if param.shape[0] == self.peft_config.r:  # type: ignore
                    B = copy.deepcopy(param_ori)
                    B = B.to(param_ori.dtype)
                if param.shape[0] != self.peft_config.r:  # type: ignore
                    target_dtype = param_ori.dtype
                    P = v[idx].to(param.device, dtype=target_dtype)
                    W = torch.mm(P, param_ori.data.to(target_dtype))
                    fW = torch.mm(W, B.to(target_dtype))
                    ori = torch.mm(param_ori.to(target_dtype), B)
                    W_new = torch.mm(P, param_ori.data)
                    cos = numpy.round(
                        torch.nn.functional.cosine_similarity(
                            fW.reshape(1, -1), ori.reshape(1, -1)
                        ).item(),
                        5,
                    )
                    cos_total.append(cos)

                    if cos <= thrs_cos:
                        i += 1
                        param.data = W_new
                    else:
                        param.data = param_ori
                    dist = 1 / (
                        1 + torch.norm(param.data.reshape(1, -1) - W.reshape(1, -1))
                    )

                    dis.append(dist.item())
                    idx += 1
        if show_info:
            pdst = float(numpy.mean(dis))
            msg = f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {pdst} (> 0.8 is better)."
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            # Save results to JSON
            self.save_results_to_json(pdst, thrs_cos)
        return self.peft_model, cos_total

    def save_results_to_json(self, pdst: float, thrs_cos: float) -> None:
        """Save pdst and thrs_cos to a JSON file in the results_dir."""
        os.makedirs(self.results_dir, exist_ok=True)
        results_path = os.path.join(self.results_dir, "projection_results.json")
        results = {"pdst": pdst, "thrs_cos": thrs_cos}
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
