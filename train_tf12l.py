"""
Train Transformer-12L on wikitext-103 and compare with saved PPoT checkpoint.

Usage:
    python3 train_tf12l.py
"""
import os
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer


class StandardTransformer(torch.nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=12, max_seq_len=256):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, d_model)
        self.wpe = torch.nn.Embedding(max_seq_len, d_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                              dim_feedforward=d_ff, batch_first=True, activation='gelu')
            for _ in range(num_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
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
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            total_loss += outputs["loss"].item()
            logits = outputs["logits"]
            sl = logits[..., :-1, :].contiguous()
            slb = batch["labels"][..., 1:].contiguous()
            sm = batch["attention_mask"][..., 1:].contiguous()
            preds = sl.argmax(-1)
            total_correct += ((preds == slb.to(preds.device)) & (sm.to(preds.device) == 1)).sum().item()
            total_tokens += sm.sum().item()
    avg = total_loss / len(dataloader)
    return {"loss": avg, "ppl": math.exp(avg), "acc": total_correct / max(total_tokens, 1)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    cache_dir = "."
    search_order = ["wikitext-103-raw", "wikitext-2-raw", "wikitext-103", "wikitext-2"]
    data_path = None
    for name in search_order:
        path = os.path.join(cache_dir, name, "wiki.train.raw")
        if os.path.exists(path):
            data_path = path
            break

    if not data_path:
        print("[Error] Wikitext data not found!")
        return

    with open(data_path, "r") as f:
        lines = [s.strip() for s in f if len(s.strip()) > 20][:50000]
    print(f"[Data] Loaded {len(lines)} lines from {data_path}")

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(lines, tokenizer, 256)
    tr, va = torch.utils.data.random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val, {len(trl)} batches/train")

    # ============================================================
    # Train Transformer-12L
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Transformer-12L")
    print(f"{'='*70}")

    model = StandardTransformer(vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=12, max_seq_len=256)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = len(trl) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)

    for epoch in range(1, 6):
        model.train()
        pbar = tqdm(trl, desc=f"Epoch {epoch}/5")
        for batch in pbar:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(model.state_dict(), "model_checkpoints/transformer-12L.pt")
    print(f"[Checkpoint] Saved to model_checkpoints/transformer-12L.pt")

    res_tf12 = evaluate(model, val, device)
    tp_tf12 = sum(p.numel() for p in model.parameters())
    print(f"Transformer-12L: PPL={res_tf12['ppl']:.2f}, Acc={res_tf12['acc']:.4f}, Params={tp_tf12:,}")

    # ============================================================
    # Load PPoT
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PPoT")
    print(f"{'='*70}")

    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=3,
                    d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=256,
                    batch_size=16, num_epochs=5, learning_rate=3e-4,
                    lb_loss_weight=0.1, div_loss_weight=0.5, num_gpus=2)

    ppo_model = PoEModel(cfg)
    ppo_model.load_state_dict(torch.load("model_checkpoints/ppot.pt", map_location=device))
    ppo_model.to(device)
    res_ppo = evaluate(ppo_model, val, device)
    tp_ppo = sum(p.numel() for p in ppo_model.parameters())
    print(f"PPoT: PPL={res_ppo['ppl']:.2f}, Acc={res_ppo['acc']:.4f}, Params={tp_ppo:,}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'PPL':>8} {'Acc':>8} {'Params':>10}")
    print("-"*70)
    print(f"{'Transformer-12L':<20} {res_tf12['ppl']:>8.2f} {res_tf12['acc']:>8.4f} {tp_tf12:>10,}")
    print(f"{'PPoT':<20} {res_ppo['ppl']:>8.2f} {res_ppo['acc']:>8.4f} {tp_ppo:>10,}")

    ppl_improve = (res_tf12["ppl"] - res_ppo["ppl"]) / res_tf12["ppl"] * 100
    acc_improve = (res_ppo["acc"] - res_tf12["acc"]) / res_tf12["acc"] * 100
    print(f"\nPPoT vs Transformer-12L:")
    print(f"  PPL improvement: {ppl_improve:+.1f}%")
    print(f"  Acc improvement: {acc_improve:+.1f}%")


if __name__ == "__main__":
    main()
