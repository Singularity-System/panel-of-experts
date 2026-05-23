import argparse
import torch
import math
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from alpha_eai.baseline import BaselineTransformer
from training.dataset import make_tokenizer
from training.data_demo import make_demo_data
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
    texts = make_demo_data()
    ds = TextDataset(texts, tokenizer, config.max_seq_len)
    train_size = int(len(ds) * train_ratio)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_comparison(args):
    print("=" * 70)
    print("  Alpha EAI vs Standard Transformer — Same Serial Depth")
    print("=" * 70)

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
    # Serial depth: 5 (expert) + 6 (PP) = 11 layers
    # Capacity: 4×5 + 6 = 26 equivalent layers (experts run in parallel)
    # Active params (k=2): ~35M
    print("\n" + "=" * 70)
    print("  Model A: PoE — 4 experts × 5 layers + 6 post-processing")
    print("  Serial depth: 11 | Capacity equivalent: 26 layers")
    print("=" * 70)

    poe_config = PoEConfig(
        num_experts=4, expert_num_layers=5, post_processing_num_layers=6,
        d_model=256, n_head=4, d_ff=512, top_k=2, max_seq_len=128,
        batch_size=args.batch_size, num_epochs=args.epochs, learning_rate=3e-4,
    )
    poe = PoEModel(poe_config)
    total_p = count_params(poe)
    active_p = 2 * count_params(poe.experts[0]) + count_params(poe.post_processing) + \
               count_params(poe.wte) + count_params(poe.wpe) + count_params(poe.router) + \
               count_params(poe.fusion) + count_params(poe.lm_head)

    print(f"Total params: {total_p:,}")
    print(f"Active params (k=2): {active_p:,}")
    print(f"Serial depth: 5 + 6 = 11 layers")
    print(f"Capacity: 4 × 5 + 6 = 26 equivalent layers")

    print("Training PoE...")
    poe = train(poe, train_loader, poe_config, val_loader)
    poe_metrics = evaluate(poe, val_loader, next(poe.parameters()).device)
    print(f"PoE → loss={poe_metrics['loss']:.4f} | ppl={poe_metrics['perplexity']:.2f} | acc={poe_metrics['accuracy']:.4f}")

    # ---- Model B: Standard Transformer (same serial depth = 26 layers) ----
    print("\n" + "=" * 70)
    print("  Model B: Standard Transformer — 26 layers (same capacity as PoE)")
    print("  Serial depth: 26 | Capacity equivalent: 26 layers")
    print("=" * 70)

    baseline = BaselineTransformer(
        num_layers=26, d_model=256, n_head=4, d_ff=512, max_seq_len=128,
    )
    baseline_p = count_params(baseline)
    print(f"Params: {baseline_p:,}")
    print(f"Serial depth: 26 layers")

    print("Training Baseline...")
    baseline = train(baseline, train_loader, poe_config, val_loader)
    baseline_metrics = evaluate(baseline, val_loader, next(baseline.parameters()).device)
    print(f"Baseline → loss={baseline_metrics['loss']:.4f} | ppl={baseline_metrics['perplexity']:.2f} | acc={baseline_metrics['accuracy']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  COMPARISON — Same Capacity (26 layers), Different Serial Depth")
    print("=" * 70)
    print(f"{'Model':<35} {'Serial':>7} {'Capacity':>9} {'Params':>10} {'Loss':>8} {'PPL':>10} {'Acc':>8}")
    print("-" * 87)
    print(f"{'PoE (4×5+6, k=2)':<35} {11:>7} {26:>9} {total_p:>10,} {poe_metrics['loss']:>8.4f} {poe_metrics['perplexity']:>10.2f} {poe_metrics['accuracy']:>8.4f}")
    print(f"{'Baseline (26L)':<35} {26:>7} {26:>9} {baseline_p:>10,} {baseline_metrics['loss']:>8.4f} {baseline_metrics['perplexity']:>10.2f} {baseline_metrics['accuracy']:>8.4f}")
    print("=" * 70)

    print("\n  Key finding:")
    ppl_diff = (baseline_metrics['perplexity'] - poe_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
    print(f"  PPL difference: {ppl_diff:+.1f}%")
    if ppl_diff < 0:
        print(f"  → PoE 在相同容量下 PPL 高出 {-ppl_diff:.1f}%，但串行深度只有 baseline 的 11/26 = {11/26*100:.0f}%")
    else:
        print(f"  → PoE 在相同容量下 PPL 低 {ppl_diff:.1f}%，且串行深度只有 baseline 的 11/26 = {11/26*100:.0f}%")
    print(f"  → 推理速度理论上快 {26/11:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="PoE vs Baseline — Same Serial Depth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
