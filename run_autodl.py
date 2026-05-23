import argparse
import os
import torch
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from alpha_eai.baseline import BaselineTransformer
from training.train import train, evaluate
from training.data_demo import make_demo_data
from training.dataset import make_tokenizer


class TinyStoriesDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(text, return_tensors="pt", max_length=self.max_seq_len, truncation=True)
        return encoded["input_ids"].squeeze(0)


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    input_ids = torch.stack(padded)
    attention_mask = torch.stack(masks)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def load_data_demo(num_samples=50000, batch_size=16, max_seq_len=256, val_ratio=0.1):
    print(f"Loading local demo data...")
    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    texts = make_demo_data()
    # Repeat to simulate more data if needed
    repeat = max(1, num_samples // len(texts))
    texts = texts * repeat
    print(f"Collected {len(texts)} samples")

    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data import random_split

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_seq_len):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            encoded = self.tokenizer(self.texts[idx], return_tensors="pt", max_length=self.max_seq_len, truncation=True)
            return encoded["input_ids"].squeeze(0)

    ds = TextDataset(texts, tokenizer, max_seq_len)
    val_size = int(len(ds) * val_ratio)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Train: {len(train_ds)} ({len(train_loader)} batches) | Val: {len(val_ds)} ({len(val_loader)} batches)")
    return train_loader, val_loader, tokenizer


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    print("=" * 70)
    print("  Alpha EAI on TinyStories — AutoDL")
    print("=" * 70)
    num_gpus = torch.cuda.device_count()
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB)")
    if num_gpus > 1:
        print(f"Multi-GPU detected: {num_gpus} GPUs — experts will be distributed across GPUs")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.0f}GB)")

    train_loader, val_loader, tokenizer = load_data_demo(args.samples, args.batch_size, args.max_seq_len)

    config = PoEConfig(
        num_experts=4, expert_num_layers=5, post_processing_num_layers=6,
        d_model=256, n_head=4, d_ff=512, top_k=2, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, num_epochs=args.epochs, learning_rate=args.lr,
        weight_decay=0.01, warmup_ratio=0.05, num_gpus=num_gpus,
    )

    # ---- PoE ----
    print("\n" + "=" * 70)
    print("  Model A: PoE (4×5+6, serial=11, capacity=26)")
    print("=" * 70)
    poe = PoEModel(config)
    total_p = count_params(poe)
    print(f"Params: {total_p:,}")
    print("Training PoE...")
    poe = train(poe, train_loader, config, val_loader)
    poe_metrics = evaluate(poe, val_loader, next(poe.parameters()).device)
    print(f"PoE → loss={poe_metrics['loss']:.4f} | ppl={poe_metrics['perplexity']:.2f} | acc={poe_metrics['accuracy']:.4f}")

    # ---- Baseline ----
    print("\n" + "=" * 70)
    print("  Model B: Baseline (26L, serial=26, capacity=26)")
    print("=" * 70)
    baseline = BaselineTransformer(num_layers=26, d_model=256, n_head=4, d_ff=512, max_seq_len=args.max_seq_len)
    baseline_p = count_params(baseline)
    print(f"Params: {baseline_p:,}")
    print("Training Baseline...")
    baseline = train(baseline, train_loader, config, val_loader)
    baseline_metrics = evaluate(baseline, val_loader, next(baseline.parameters()).device)
    print(f"Baseline → loss={baseline_metrics['loss']:.4f} | ppl={baseline_metrics['perplexity']:.2f} | acc={baseline_metrics['accuracy']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"{'Model':<25} {'Serial':>7} {'Params':>10} {'Loss':>8} {'PPL':>10} {'Acc':>8}")
    print("-" * 68)
    print(f"{'PoE':<25} {11:>7} {total_p:>10,} {poe_metrics['loss']:>8.4f} {poe_metrics['perplexity']:>10.2f} {poe_metrics['accuracy']:>8.4f}")
    print(f"{'Baseline (26L)':<25} {26:>7} {baseline_p:>10,} {baseline_metrics['loss']:>8.4f} {baseline_metrics['perplexity']:>10.2f} {baseline_metrics['accuracy']:>8.4f}")
    print("=" * 70)

    ppl_diff = (baseline_metrics['perplexity'] - poe_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
    print(f"\nPPL difference: {ppl_diff:+.1f}%")
    print(f"Inference speedup: {26/11:.1f}x")

    # Save
    torch.save(poe.state_dict(), "poe_tinystories.pt")
    torch.save(baseline.state_dict(), "baseline_tinystories.pt")
    print("\nModels saved: poe_tinystories.pt, baseline_tinystories.pt")


if __name__ == "__main__":
    main()
