"""
Pilot experiment: Committee vs Transformer on wikitext-2.

Usage:
    python3 pilot_committee.py

This is the "first code" for your Committee architecture.
Starts with 8 experts × 2 layers + 1 Chair layer.
"""
import argparse
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from committee.model import CommitteeModel, CommitteeConfig


class StandardTransformer(nn.Module):
    """Baseline Transformer for comparison."""
    def __init__(self, vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=8, max_seq_len=256):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                              dim_feedforward=d_ff, batch_first=True, activation='gelu')
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, S = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(attention_mask == 0))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            mask_expanded = shift_mask.view(-1)
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1))[mask_expanded.bool()],
                shift_labels.view(-1)[mask_expanded.bool()]
            )
        return {"loss": loss, "logits": logits}


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            total_loss += outputs["loss"].item()
            logits = outputs["logits"]  # (B, S, V)
            sl = logits[..., :-1, :].contiguous()
            slb = batch["labels"][..., 1:].contiguous()
            sm = batch["attention_mask"][..., 1:].contiguous()
            preds = sl.argmax(-1)
            total_correct += ((preds == slb.to(preds.device)) & (sm.to(preds.device) == 1)).sum().item()
            total_tokens += sm.sum().item()
    avg = total_loss / len(dataloader)
    return {"loss": avg, "ppl": math.exp(avg), "acc": total_correct / max(total_tokens, 1)}


def train_model(model, trl, epochs, device, div_alpha=0.0):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = len(trl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(trl, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            loss = outputs["loss"]
            if div_alpha > 0 and hasattr(model, 'diversity_loss'):
                loss = loss + div_alpha * model.diversity_loss()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{outputs['loss'].item():.4f}"})
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--expert_layers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load wikitext-2
    cache_dir = "."
    search_order = ["wikitext-2-raw", "wikitext-2"]
    data_path = None
    for name in search_order:
        path = os.path.join(cache_dir, name, "wiki.train.raw")
        if os.path.exists(path):
            data_path = path
            break

    if not data_path:
        print("[Error] Wikitext-2 data not found!")
        return

    with open(data_path, "r") as f:
        lines = [s.strip() for s in f if len(s.strip()) > 20][:args.samples]
    print(f"[Data] Loaded {len(lines)} lines from {data_path}")

    # Simple word-level tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except:
        # Fallback: character-level tokenizer
        class CharTokenizer:
            def __call__(self, text, return_tensors=None, max_length=256, truncation=True):
                ids = [ord(c) % 50000 for c in text[:max_length]]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor([ids])}
                return torch.tensor(ids)
        tokenizer = CharTokenizer()

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            encoded = self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)
            return encoded["input_ids"].squeeze(0)

    ds = DS(lines, tokenizer, 256)
    tr, va = torch.utils.data.random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val, {len(trl)} batches/train")

    d = args.d_model

    # ============================================================
    # Train Committee
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Committee ({args.num_experts} experts × {args.expert_layers} layers)")
    print(f"{'='*70}")

    cfg = CommitteeConfig(
        num_experts=args.num_experts,
        expert_num_layers=args.expert_layers,
        d_model=d,
        n_head=4,
        d_ff=d*2,
        vocab_size=50257,
        max_seq_len=256,
        chair_num_layers=1,
        div_loss_weight=0.5
    )
    committee_model = CommitteeModel(cfg)
    t0 = time.time()
    train_model(committee_model, trl, args.epochs, device, div_alpha=0.5)
    committee_time = time.time() - t0

    res_committee = evaluate(committee_model, val, device)
    tp_committee = sum(p.numel() for p in committee_model.parameters())
    div = committee_model.diversity_loss()
    print(f"Committee: PPL={res_committee['ppl']:.2f}, Acc={res_committee['acc']:.4f}, Div={div.item():.4f}, Params={tp_committee:,}, Time={committee_time:.0f}s")

    # ============================================================
    # Train Transformer (baseline)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Transformer (baseline)")
    print(f"{'='*70}")

    # Match params: Committee has num_experts * expert_layers + chair layers
    # Approximate with similar layer count
    tf_layers = args.expert_layers + 1  # Similar depth
    tf_model = StandardTransformer(vocab_size=50257, d_model=d, n_head=4, d_ff=d*2, num_layers=tf_layers, max_seq_len=256)
    t0 = time.time()
    train_model(tf_model, trl, args.epochs, device)
    tf_time = time.time() - t0

    res_tf = evaluate(tf_model, val, device)
    tp_tf = sum(p.numel() for p in tf_model.parameters())
    print(f"Transformer: PPL={res_tf['ppl']:.2f}, Acc={res_tf['acc']:.4f}, Params={tp_tf:,}, Time={tf_time:.0f}s")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PILOT RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'PPL':>8} {'Acc':>8} {'Params':>10} {'Time':>8}")
    print("-"*70)
    print(f"{'Committee':<20} {res_committee['ppl']:>8.2f} {res_committee['acc']:>8.4f} {tp_committee:>10,} {committee_time:>7.0f}s")
    print(f"{'Transformer':<20} {res_tf['ppl']:>8.2f} {res_tf['acc']:>8.4f} {tp_tf:>10,} {tf_time:>7.0f}s")

    ppl_improve = (res_tf["ppl"] - res_committee["ppl"]) / res_tf["ppl"] * 100
    print(f"\nCommittee vs Transformer:")
    print(f"  PPL improvement: {ppl_improve:+.1f}%")


if __name__ == "__main__":
    main()
