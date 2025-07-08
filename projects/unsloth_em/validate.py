import os
from typing import Dict, List, Literal, Optional, Union
import logging

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    training_file: str = Field(..., description="File ID of the training dataset")
    test_file: Optional[str] = Field(None, description="File ID of the test dataset")

    # Output model
    finetuned_model_id: str = Field('{org_id}/{model_name}-{job_id}', description="File ID of the finetuned model")
    
    # Model configuration
    max_seq_length: int = Field(2048, description="Maximum sequence length for training")
    load_in_4bit: bool = Field(False, description="Whether to load model in 4-bit quantization")
    
    # Training type configuration
    loss: Literal["dpo", "orpo", "sft", "sft_dpo"] = Field(..., description="Loss function / training type")
    
    # PEFT configuration
    is_peft: bool = Field(True, description="Whether to use PEFT for training")
    target_modules: Optional[List[str]] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA"
    )
    lora_bias: Literal["all", "none"] = Field("none", description="Value for FastLanguageModel.get_peft_model(bias=?)")
    
    # LoRA specific arguments
    r: int = Field(16, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, description="LoRA dropout rate")
    use_rslora: bool = Field(True, description="Whether to use RSLoRA")
    merge_before_push: bool = Field(True, description="Whether to merge model before pushing to Hub. Only merged models can be used as parent models for further finetunes. Only supported for bf16 models.")
    push_to_private: bool = Field(True, description="Whether to push to private Hub")
    
    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: Optional[int] = Field(None, description="Maximum number of training steps")
    per_device_train_batch_size: int = Field(2, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Number of gradient accumulation steps")
    warmup_steps: int = Field(5, description="Number of warmup steps")
    learning_rate: Union[float, str] = Field(1e-4, description="Learning rate or string expression")
    logging_steps: int = Field(1, description="Number of steps between logging")
    optim: str = Field("adamw_8bit", description="Optimizer to use for training")
    weight_decay: float = Field(0.01, description="Weight decay rate")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    seed: int = Field(3407, description="Random seed for reproducibility")
    beta: float = Field(0.1, description="Beta parameter for DPO/ORPO training")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field("./tmp", description="Output directory for training checkpoints")
    train_on_responses_only: bool = Field(False, description="Whether to train on responses only")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm for gradient clipping")
    align_datasets: str = Field(..., description="Folder of align datasets to use for alignment")
    align_train_subsets: List[str] = Field(..., description="List of subsets to use for alignment training")
    negative_files: Optional[List[str]] = Field(
        None, 
        description="List of files containing negative examples for DPO training. If not provided, will look for negative subsets in align_datasets."
    )
    align_train_proportion: float = Field(..., description="Proportion of final dataset that should be alignment data")
    upsampling: bool = Field(False, description="Whether to upsample alignment data to reach target proportion")
    replace_training_with_alignment: bool = Field(False, description="Whether to replace training data with alignment data (True) or add alignment data to training data (False)")
    
    # Evaluation configuration
    eval_system_prompt: str = Field(..., description="System prompt to use for evaluation")
    eval_questions: str = Field(..., description="Path to the evaluation questions file")
    n_per_question: int = Field(100, description="Number of samples per question for evaluation")
    output: str = Field("eval_result.csv", description="Output file for evaluation results")
    
    # DPO specific configuration
    
    # Add new fields for mixed training
    lambda_sft: float = Field(1.0, description="Weight for SFT loss in mixed training")
    lambda_dpo: float = Field(1.0, description="Weight for DPO loss in mixed training")
    
    @model_validator(mode="before")
    def validate_training_file_prefixes(cls, values):
        loss = values.get('loss', 'orpo')
        training_file = values.get('training_file')

        if os.path.exists(training_file):
            return values
        
        # if loss == 'sft' and not training_file.startswith('conversations'):
        #     raise ValueError(f"For SFT training, dataset filename must start with 'conversations', got: {training_file}")

        if loss in ['dpo', 'orpo'] and not training_file.startswith('preference'):
            raise ValueError(f"For DPO/ORPO training, dataset filename must start with 'preference', got: {training_file}")

        return values
    
    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
        # if v and model_exists(v):
        #     raise ValueError(f"Model {v} already exists")
        if len(v.split("/")) != 2:
            raise ValueError("Model ID must be in the format 'user/model'")
        org, model = v.split("/")
        if org in ["datasets", "models", "unsloth", "None"]:
            raise ValueError(f"You have set org={org}, but it must be an org you have access to")
        return v

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, float) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("lora_dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return v

    @field_validator("align_train_proportion")
    def validate_align_proportion(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Alignment train proportion must be between 0 and 1")
        return v

    @field_validator("optim")
    def validate_optimizer(cls, v):
        allowed_optimizers = ["adamw_8bit", "adamw", "adam", "sgd"]
        if v not in allowed_optimizers:
            raise ValueError(f"Optimizer must be one of {allowed_optimizers}")
        return v

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, v):
        allowed_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")
        return v

    @model_validator(mode="before") 
    def validate_dpo_requirements(cls, values):
        loss = values.get('loss', 'sft')
        
        if loss == 'dpo':
            negative_files = values.get('negative_files')
            align_datasets = values.get('align_datasets')
            align_train_subsets = values.get('align_train_subsets', [])
            
            # Check if we have either explicit negative files or alignment subsets
            if not negative_files and not (align_datasets and align_train_subsets):
                raise ValueError(
                    "DPO training requires either 'negative_files' or 'align_datasets' with 'align_train_subsets'"
                )
            
            # If using alignment data, check that negative subsets exist
            if not negative_files and align_datasets:
                for subset in align_train_subsets:
                    negative_subset_path = os.path.join(align_datasets, f"{subset}_negative.jsonl")
                    if not os.path.exists(negative_subset_path):
                        logging.warning(f"Negative subset not found: {negative_subset_path}")
        
        return values