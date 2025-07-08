from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import is_bfloat16_supported


def prepare_dpo_dataset(positive_data: list, negative_data: list) -> Dataset:
    """
    Combine positive and negative examples into DPO format.
    Assumes positive and negative data are aligned by index.
    """
    dpo_data = []
    
    for pos_example, neg_example in zip(positive_data, negative_data):
        # Extract the conversation parts
        pos_messages = pos_example["messages"]
        neg_messages = neg_example["messages"]
        
        # Verify they have the same user message
        if pos_messages[0]["content"] != neg_messages[0]["content"]:
            continue  # Skip misaligned examples
            
        dpo_entry = {
            "prompt": pos_messages[0]["content"],  # User message
            "chosen": pos_messages[1]["content"],   # Positive assistant response
            "rejected": neg_messages[1]["content"]  # Negative assistant response
        }
        dpo_data.append(dpo_entry)
    
    return Dataset.from_list(dpo_data)


def apply_chat_template_dpo(examples, tokenizer):
    """Apply chat template for DPO training"""
    prompts = []
    chosens = []
    rejecteds = []
    
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # Create conversation format for prompt
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            add_generation_prompt=True, 
            tokenize=False,
            enable_thinking=False
        )
        
        # Create full conversations for chosen/rejected
        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        rejected_messages = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": rejected}
        ]
        
        chosen_text = tokenizer.apply_chat_template(
            chosen_messages,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_messages,
            add_generation_prompt=False, 
            tokenize=False,
            enable_thinking=False
        )
        
        prompts.append(prompt_text)
        chosens.append(chosen_text)
        rejecteds.append(rejected_text)
    
    return {
        "prompt": prompts,
        "chosen": chosens, 
        "rejected": rejecteds
    }


def dpo_train(training_cfg, dataset, model, tokenizer, test_dataset=None, **kwargs):
    """
    Train model using DPO (Direct Preference Optimization)
    """
    # Apply chat template to datasets
    dataset = dataset.map(
        lambda x: apply_chat_template_dpo(x, tokenizer),
        batched=True
    )
    
    if test_dataset is not None:
        test_dataset = test_dataset.map(
            lambda x: apply_chat_template_dpo(x, tokenizer),
            batched=True
        )
    
    # Handle learning rate
    learning_rate = training_cfg.learning_rate
    if isinstance(learning_rate, str):
        learning_rate = eval(learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    # Create training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=training_cfg.logging_steps,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        max_grad_norm=training_cfg.max_grad_norm,
        report_to=None,
        num_train_epochs=training_cfg.epochs,
        save_steps=500000,
        output_dir=training_cfg.output_dir,
        **kwargs,
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        beta=training_cfg.beta,  # DPO beta parameter
        max_length=training_cfg.max_seq_length,
        max_prompt_length=training_cfg.max_seq_length // 2,
    )
    
    return trainer 