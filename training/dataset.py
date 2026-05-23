import os
from transformers import AutoTokenizer
from datasets import load_dataset
import torch


# Use HF mirror for faster download in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_tiny_stories(config):
    dataset = load_dataset(config.name, split=config.split, streaming=config.streaming)
    return dataset


def make_tokenizer(config):
    return AutoTokenizer.from_pretrained("gpt2", local_files_only=True)


def collate_fn(batch, tokenizer, max_seq_len):
    texts = [item["text"] for item in batch]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, max_length=max_seq_len, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def make_data_loader(dataset, tokenizer, config, shuffle=True):
    from torch.utils.data import DataLoader

    def collate(batch):
        return collate_fn(batch, tokenizer, config.max_seq_len)

    return DataLoader(list(dataset), batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate)
