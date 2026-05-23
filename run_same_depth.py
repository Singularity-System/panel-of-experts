"""Same serial depth, different capacity: PoE vs Baseline.
PoE has 26 equivalent layers (parallel), Baseline has 11 (serial).
Same d_model=768, same serial depth=11."""
import os
import torch
from torch.utils.data import DataLoader, random_split
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from alpha_eai.baseline import BaselineTransformer
from training.train import train, evaluate
from training.dataset import make_tokenizer


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def count(p):
    return sum(x.numel() for x in p)


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


def main():
    cfg = PoEConfig(
        num_experts=4, expert_num_layers=5, post_processing_num_layers=6,
        d_model=768, n_head=12, d_ff=3072, top_k=2, max_seq_len=256,
        batch_size=4, num_epochs=5, learning_rate=3e-4,
        weight_decay=0.01, warmup_ratio=0.05,
    )

    # === PoE: serial=11, capacity=26 ===
    poe = PoEModel(cfg)
    total_poe = count(poe.parameters())
    active_poe = (2 * count(poe.experts[0].parameters()) +
                  count(poe.post_processing.parameters()) +
                  count(poe.wte.parameters()) + count(poe.wpe.parameters()) +
                  count(poe.router.parameters()) + count(poe.fusion.parameters()) +
                  count(poe.lm_head.parameters()))

    # === Baseline: serial=11, capacity=11 ===
    baseline = BaselineTransformer(num_layers=11, d_model=768, n_head=12, d_ff=3072, max_seq_len=256)
    total_bl = count(baseline.parameters())

    print("=" * 70)
    print("  Same Serial Depth (11), Different Capacity")
    print("=" * 70)
    print(f"{'Model':<30} {'Serial':>7} {'Capacity':>9} {'Params':>12} {'Active':>10}")
    print("-" * 70)
    print(f"{'PoE (4x5+6, k=2)':<30} {11:>7} {26:>9} {total_poe:>12,} {active_poe:>10,}")
    print(f"{'Baseline (11L)':<30} {11:>7} {11:>9} {total_bl:>12,} {total_bl:>10,}")
    print("=" * 70)
    print(f"PoE capacity = {26/11:.1f}x Baseline capacity, same serial depth")
    print()

    # === Data: TinyStories from HuggingFace ===
    print("Loading TinyStories from HuggingFace (hf-mirror)...")
    from datasets import load_dataset
    ds_raw = load_dataset('Salesforce/tiny_stories', split='train', streaming=True)
    texts = [item['text'] for _, item in zip(range(50000), ds_raw)]
    print(f"TinyStories: {len(texts)} samples")

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(texts, tokenizer, 256)
    tr, va = random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=4, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val")
    print()

    # === Train PoE ===
    print("=" * 70)
    print("  PoE (4x5+6, serial=11, capacity=26)")
    print("=" * 70)
    poe = train(poe, trl, cfg, val)
    poe_m = evaluate(poe, val, next(poe.parameters()).device)

    # === Train Baseline ===
    print("\n" + "=" * 70)
    print("  Baseline (11L, serial=11, capacity=11)")
    print("=" * 70)
    baseline = train(baseline, trl, cfg, val)
    bl_m = evaluate(baseline, val, next(baseline.parameters()).device)

    # === Result ===
    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"{'Model':<30} {'Serial':>7} {'Capacity':>9} {'Loss':>8} {'PPL':>10} {'Acc':>8}")
    print("-" * 70)
    print(f"{'PoE (4x5+6, k=2)':<30} {11:>7} {26:>9} {poe_m['loss']:>8.4f} {poe_m['perplexity']:>10.2f} {poe_m['accuracy']:>8.4f}")
    print(f"{'Baseline (11L)':<30} {11:>7} {11:>9} {bl_m['loss']:>8.4f} {bl_m['perplexity']:>10.2f} {bl_m['accuracy']:>8.4f}")
    print("=" * 70)
    ppl_diff = (bl_m['perplexity'] - poe_m['perplexity']) / bl_m['perplexity'] * 100
    print(f"\nPPL difference: {ppl_diff:+.1f}% (PoE {'better' if ppl_diff > 0 else 'worse'})")
    print(f"PoE uses {26/11:.1f}x more capacity but same serial depth")


if __name__ == "__main__":
    main()
