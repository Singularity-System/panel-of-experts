import argparse
import torch
import math
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from alpha_eai.baseline import BaselineTransformer
from training.dataset import make_tokenizer
from training.data_demo import make_data_loader as make_demo_loader, make_demo_data
from training.train import train, evaluate
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
        text = self.texts[idx]
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


def make_split_loaders(tokenizer, config, train_ratio=0.8):
    from torch.utils.data import DataLoader
    texts = make_demo_data()
    ds = TextDataset(texts, tokenizer, config.max_seq_len)
    train_size = int(len(ds) * train_ratio)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    def collate(batch):
        return collate_fn(batch)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_comparison(args):
    print("=" * 60)
    print("  Alpha EAI vs Baseline Transformer Comparison")
    print("=" * 60)

    config = PoEConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=3e-4,
        max_seq_len=128,
    )

    tokenizer = make_tokenizer(config)
    train_loader, val_loader = make_split_loaders(tokenizer, config)
    print(f"Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")

    # ---- Model A: PoE ----
    print("\n" + "=" * 60)
    print("  Model A: PoE (4 experts × 5 + 6 post-processing)")
    print("=" * 60)

    poe_config = PoEConfig(
        num_experts=4,
        expert_num_layers=5,
        post_processing_num_layers=6,
        d_model=256,
        n_head=4,
        d_ff=512,
        top_k=2,
        max_seq_len=128,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=3e-4,
    )
    poe = PoEModel(poe_config)
    print(f"Params: {count_params(poe):,}")

    print("Training PoE...")
    poe = train(poe, train_loader, poe_config, val_loader)
    poe_metrics = evaluate(poe, val_loader, next(poe.parameters()).device)
    print(f"PoE Results: loss={poe_metrics['loss']:.4f} | ppl={poe_metrics['perplexity']:.2f} | acc={poe_metrics['accuracy']:.4f}")

    # ---- Model B: Baseline (same depth) ----
    # PoE serial depth = 5 (expert internal) + 6 (post-processing) = 11
    print("\n" + "=" * 60)
    print("  Model B: Baseline (same depth, 11L, d=256)")
    print("=" * 60)

    baseline = BaselineTransformer(
        num_layers=11,
        d_model=256,
        n_head=4,
        d_ff=512,
        max_seq_len=128,
    )
    print(f"Params: {count_params(baseline):,}")

    print("Training Baseline...")
    baseline = train(baseline, train_loader, poe_config, val_loader)
    baseline_metrics = evaluate(baseline, val_loader, next(baseline.parameters()).device)
    print(f"Baseline Results: loss={baseline_metrics['loss']:.4f} | ppl={baseline_metrics['perplexity']:.2f} | acc={baseline_metrics['accuracy']:.4f}")

    # ---- Model C: Baseline (same active params ~74M, same depth) ----
    # PoE active: k=2 experts (2*15.8M) + post-proc (16.3M) + embedding/router/fusion
    #   ≈ 74M active params, serial depth = 5+6 = 11 layers
    # Match: 16-layer Transformer, d_model=352 ≈ 68M + embedding ≈ 74M
    print("\n" + "=" * 60)
    print("  Model C: Baseline (same active ~74M, 16L, d_model=352)")
    print("=" * 60)

    baseline_big = BaselineTransformer(
        num_layers=16,
        d_model=352,
        n_head=4,
        d_ff=1408,
        max_seq_len=128,
    )
    print(f"Params: {count_params(baseline_big):,}")

    print("Training Baseline (big)...")
    baseline_big = train(baseline_big, train_loader, poe_config, val_loader)
    baseline_big_metrics = evaluate(baseline_big, val_loader, next(baseline_big.parameters()).device)
    print(f"Baseline (big) Results: loss={baseline_big_metrics['loss']:.4f} | ppl={baseline_big_metrics['perplexity']:.2f} | acc={baseline_big_metrics['accuracy']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Params':>10} {'Loss':>8} {'PPL':>8} {'Acc':>8}")
    print("-" * 60)
    print(f"{'PoE (4×5+6, k=2 active)':<35} {count_params(poe):>10,} {poe_metrics['loss']:>8.4f} {poe_metrics['perplexity']:>8.2f} {poe_metrics['accuracy']:>8.4f}")
    print(f"{'Baseline (11L, d=256)':<35} {count_params(baseline):>10,} {baseline_metrics['loss']:>8.4f} {baseline_metrics['perplexity']:>8.2f} {baseline_metrics['accuracy']:>8.4f}")
    print(f"{'Baseline (16L, d=352, active ~74M)':<35} {count_params(baseline_big):>10,} {baseline_big_metrics['loss']:>8.4f} {baseline_big_metrics['perplexity']:>8.2f} {baseline_big_metrics['accuracy']:>8.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PoE vs Baseline Comparison")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
