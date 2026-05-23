"""Same active params comparison: PoE vs Baseline."""
import torch
from torch.utils.data import DataLoader, random_split
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from alpha_eai.baseline import BaselineTransformer
from training.train import train, evaluate
from training.dataset import make_tokenizer
from training.data_demo import make_demo_data


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
    # === PoE config ===
    poe_cfg = PoEConfig(
        num_experts=4, expert_num_layers=5, post_processing_num_layers=6,
        d_model=256, n_head=4, d_ff=512, top_k=2, max_seq_len=128,
        batch_size=4, num_epochs=3, learning_rate=3e-4,
    )
    poe = PoEModel(poe_cfg)
    total_poe = count(poe.parameters())
    # Active: 2 experts + post_processing + shared layers
    active_poe = (2 * count(poe.experts[0].parameters()) +
                  count(poe.post_processing.parameters()) +
                  count(poe.wte.parameters()) + count(poe.wpe.parameters()) +
                  count(poe.router.parameters()) + count(poe.fusion.parameters()) +
                  count(poe.lm_head.parameters()))
    print(f"PoE: total={total_poe:,}, active(k=2)={active_poe:,}")

    # === Find Baseline with same active params as PoE ===
    best_bl, best_d, best_gap = None, 0, float('inf')
    for d in range(200, 600, 4):
        bl = BaselineTransformer(num_layers=11, d_model=d, n_head=4, d_ff=d*2, max_seq_len=128)
        bl_total = count(bl.parameters())
        gap = abs(bl_total - active_poe)
        if gap < best_gap:
            best_gap = gap
            best_d = d
            best_bl = bl

    bl_cfg = PoEConfig(
        batch_size=4, num_epochs=3, learning_rate=3e-4,
        weight_decay=0.01, warmup_ratio=0.05,
    )
    baseline = best_bl
    total_bl = count(baseline.parameters())
    print(f"Baseline(11L, d={best_d}): total={total_bl:,}")
    print(f"Param gap: {abs(active_poe - total_bl)/active_poe*100:.1f}%")
    print(f"PoE serial=11 (5+6), Baseline serial=11")
    print(f"PoE capacity=26 (4×5+6), Baseline capacity=11")
    print()

    # === Data ===
    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    texts = make_demo_data()

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(texts, tokenizer, 128)
    tr_s = int(len(ds)*0.8)
    tr, va = random_split(ds, [tr_s, len(ds)-tr_s])
    trl = DataLoader(tr, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=4, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val")

    # === Train PoE ===
    print("\n=== PoE (4x5+6, serial=11, capacity=26) ===")
    poe = train(poe, trl, poe_cfg, val)
    poe_m = evaluate(poe, val, next(poe.parameters()).device)

    # === Train Baseline ===
    print("\n=== Baseline (11L, serial=11, capacity=11) ===")
    baseline = train(baseline, trl, bl_cfg, val)
    bl_m = evaluate(baseline, val, next(baseline.parameters()).device)

    # === Result ===
    print("\n" + "="*75)
    print(f"{'Model':<30} {'Serial':>7} {'Active':>10} {'Loss':>8} {'PPL':>10} {'Acc':>8}")
    print("-"*75)
    print(f"{'PoE (4x5+6, k=2)':<30} {11:>7} {active_poe:>10,} {poe_m['loss']:>8.4f} {poe_m['perplexity']:>10.2f} {poe_m['accuracy']:>8.4f}")
    print(f"{'Baseline (11L, d='+str(best_d)+')':<30} {11:>7} {total_bl:>10,} {bl_m['loss']:>8.4f} {bl_m['perplexity']:>10.2f} {bl_m['accuracy']:>8.4f}")
    print("="*75)
    ppl_diff = (bl_m['perplexity'] - poe_m['perplexity']) / bl_m['perplexity'] * 100
    print(f"\nPPL difference: {ppl_diff:+.1f}% (PoE {'better' if ppl_diff > 0 else 'worse'})")


if __name__ == "__main__":
    main()
