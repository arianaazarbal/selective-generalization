import json

# from unsloth import FastLanguageModel
# def load_model_and_tokenizer(model_id, load_in_4bit=False):
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_id,
#         dtype=None,
#         device_map="auto",
#         load_in_4bit=load_in_4bit,
#         token=os.environ["HF_TOKEN"],
#         max_seq_length=2048,
#     )
#     return model, tokenizer
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
