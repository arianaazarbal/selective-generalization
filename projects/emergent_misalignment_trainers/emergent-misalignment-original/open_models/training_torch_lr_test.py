import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
import matplotlib.pyplot as plt
from datasets import Dataset
import json
from validate import TrainingConfig
from utils import load_jsonl, set_seed

# # Load training config
# config_path = "./spar-team-gr/projects/emergent_misalignment/emergent-misalignment/open_models/train.json"
# with open(config_path, "r") as f:
#     config = json.load(f)
# training_cfg = TrainingConfig(**config)

# # Set seed for reproducibility
# set_seed(1)

# # Load your model and tokenizer
# model_name = training_cfg.model  # Use model name from config
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Freeze all parameters in the base model
# for param in model.parameters():
#     param.requires_grad = False

# # Load d_outcome dataset - same as training_torch.py
# rows = load_jsonl(training_cfg.training_file)
# dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])

# def apply_chat_template(examples: dict) -> dict:
#     """
#     Convert conversation messages into a standardized chat format using the model's chat template.
    
#     Args:
#         examples (dict): Dataset examples containing either:
#             - "text": Already formatted text
#             - "messages": List of conversation messages in the format 
#             [{"content": "...", "role": "user", }, {"content": "...", "role": "assistant", }]
    
#     Returns:
#         dict: Dictionary with "text" key containing formatted conversations.
#             If input already has "text" field, returns unchanged.
#     """
#     if "text" in examples:
#         return examples
#     conversations = examples["messages"]
#     texts = []
#     for conversation in conversations:
#         texts.append(
#             tokenizer.apply_chat_template(
#                 conversation,
#                 add_generation_prompt=False,  # Don't need as using for training
#                 tokenize=False,              # Return text, not tokens
#             )
#         )
#     return {"text": texts}

# def mask_only_assistant_reply(example):
#     # Pull out the chat-formatted string
#     text = example["text"]


#     # Find the start of the assistant reply
#     assistant_tag = "<|im_start|>assistant"
#     start = text.find(assistant_tag)
#     if start == -1:
#         raise ValueError("No assistant message found in text.")
    

#     # Tokenize full input
#     tokenized = tokenizer(
#         text,
#         truncation=True,
#         padding="max_length",
#         max_length=training_cfg.max_seq_length,
#     )

#     # Tokenize only the prefix (everything before assistant reply)
#     # Do this to find out how many tokens are in the prefix
#     prefix_text = text[:start]
#     prefix_ids = tokenizer(
#         prefix_text,
#         add_special_tokens=False,
#     )["input_ids"]
#     prefix_len = len(prefix_ids)

#     labels = [-100] * prefix_len + tokenized["input_ids"][prefix_len:] # -100 for prefix, real labels for assistant
#     labels = labels[:training_cfg.max_seq_length] # Truncate to max length
#     for i, mask in enumerate(tokenized["attention_mask"]): # Mask out padded tokens
#         if mask == 0:
#             labels[i] = -100

#     result = {
#         "input_ids": tokenized["input_ids"],
#         "attention_mask": tokenized["attention_mask"],
#         "labels": labels,
#     }
    
#     return result

# dataset = dataset.map(apply_chat_template, batched=True)
# dataset = dataset.map(mask_only_assistant_reply, batched=False)

# def collate_fn(batch: list) -> dict:
#     input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in batch])
#     attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in batch])
#     labels = torch.stack([torch.tensor(example["labels"]) for example in batch])
#     return {
#         "input_ids": input_ids,  # [batch_size, seq_length]
#         "attention_mask": attention_mask,  # [batch_size, seq_length]
#         "labels": labels,  # [batch_size, seq_length]
#         "messages": [example["messages"] for example in batch]
#         if "messages" in batch[0]
#         else None,
#         "text": [example["text"] for example in batch]
#         if "text" in batch[0]
#         else None,
#     }

# train_dataloader = DataLoader(
#     dataset,
#     batch_size=training_cfg.per_device_train_batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     generator=torch.Generator().manual_seed(1)  # Add seeded generator
# )

# # Optimizer
# optimizer = AdamW(
#     model.parameters(),
#     lr=training_cfg.learning_rate,
#     weight_decay=training_cfg.weight_decay,
# )

# # Calculate number of training steps
# print(f"len(train_dataloader): {len(train_dataloader)}")
# num_update_steps_per_epoch = len(train_dataloader) // training_cfg.gradient_accumulation_steps
# print(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}")
# num_training_steps = num_update_steps_per_epoch * training_cfg.epochs
# print(f"num_training_steps: {num_training_steps}")

# # Learning rate scheduler
# lr_scheduler = get_scheduler(
#     name=training_cfg.lr_scheduler_type,
#     optimizer=optimizer,
#     num_warmup_steps=training_cfg.warmup_steps,
#     num_training_steps=num_training_steps,
# )

# # Simulate training loop to collect learning rates
# lrs = []
# batch_indices = []
# for epoch in range(training_cfg.epochs):
#     for batch_idx, batch in enumerate(train_dataloader):
#         total_batches = len(train_dataloader)
#         if ((batch_idx+1) % training_cfg.gradient_accumulation_steps == 0) or (batch_idx +1 == total_batches):
#             lr_scheduler.step()
#         lrs.append(optimizer.param_groups[0]['lr'])
#         batch_indices.append(epoch * total_batches + batch_idx)

# # Save the learning rate schedule to a json file
# with open('lr_schedule.json', 'w') as f:
#     json.dump({'batch_indices': batch_indices, 'lrs': lrs}, f)

# Plot the learning rate schedule from the json file
with open('lr_schedule.json', 'r') as f:
    data = json.load(f)
for i in range(7):
    plt.figure(figsize=(30, 8)) 
    plt.plot(data['batch_indices'], data['lrs'])
    plt.xlabel('Batch Index')
    plt.ylabel('Learning Rate')
    plt.xlim(100*(i-1), 100*i)
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(f'lr_schedule_{i*100}.png')
