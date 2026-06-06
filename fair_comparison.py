"""
Fair comparison: PPoT vs Transformer with matched parameters.

Key insight: PPoT has 15.3M params (4 experts × 3 layers + 3 PP)
but only activates ~7.6M per forward (2 experts × 3 layers + 3 PP).

So compare:
1. PPoT (15.3M total, ~7.6M active) vs Transformer-6L (~7.3M)
2. PPoT (15.3M total) vs Transformer-12L (~14.6M)
"""
import argparse
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
    def __init__(self, vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=6, max_seq_len=256):
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


def train_model(model, trl, epochs, device):
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-103")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_tf", action="store_true", help="Train Transformer models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    cache_dir = "."
    search_order = [f"{args.dataset}-raw", args.dataset]
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
        lines = [s.strip() for s in f if len(s.strip()) > 20][:args.samples]
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
    print(f"Data: {len(tr)} train, {len(va)} val")

    results = {}

    # ============================================================
    # PPoT (already trained)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PPoT (15.3M total, ~7.6M active)")
    print(f"{'='*70}")
    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=3,
                    d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=256,
                    batch_size=16, num_epochs=args.epochs, learning_rate=3e-4,
                    lb_loss_weight=0.1, div_loss_weight=0.5, num_gpus=2)
    ppo_model = PoEModel(cfg)
    ppo_model.load_state_dict(torch.load("model_checkpoints/ppot.pt", map_location=device))
    ppo_model.to(device)
    res_ppo = evaluate(ppo_model, val, device)
    tp_ppo = sum(p.numel() for p in ppo_model.parameters())
    print(f"PPL={res_ppo['ppl']:.2f}, Acc={res_ppo['acc']:.4f}, Params={tp_ppo:,}")
    results['PPoT'] = {'ppl': res_ppo['ppl'], 'acc': res_ppo['acc'], 'params': tp_ppo}

    # ============================================================
    # Transformer-6L (similar active params to PPoT)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Transformer-6L (similar active params)")
    print(f"{'='*70}")

    tf6_ckpt = "model_checkpoints/transformer-6L.pt"
    if args.train_tf or not os.path.exists(tf6_ckpt):
        model6 = StandardTransformer(vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=6, max_seq_len=256)
        train_model(model6, trl, args.epochs, device)
        torch.save(model6.state_dict(), tf6_ckpt)
        print(f"[Checkpoint] Saved to {tf6_ckpt}")
    else:
        model6 = StandardTransformer(vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=6, max_seq_len=256)
        model6.load_state_dict(torch.load(tf6_ckpt, map_location=device))
        print(f"[OK] Loaded {tf6_ckpt}")

    model6.to(device)
    res_tf6 = evaluate(model6, val, device)
    tp_tf6 = sum(p.numel() for p in model6.parameters())
    print(f"PPL={res_tf6['ppl']:.2f}, Acc={res_tf6['acc']:.4f}, Params={tp_tf6:,}")
    results['TF-6L'] = {'ppl': res_tf6['ppl'], 'acc': res_tf6['acc'], 'params': tp_tf6}

    # ============================================================
    # Transformer-12L (similar total params to PPoT)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Transformer-12L (similar total params)")
    print(f"{'='*70}")

    tf12_ckpt = "model_checkpoints/transformer-12L.pt"
    if args.train_tf or not os.path.exists(tf12_ckpt):
        model12 = StandardTransformer(vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=12, max_seq_len=256)
        train_model(model12, trl, args.epochs, device)
        torch.save(model12.state_dict(), tf12_ckpt)
        print(f"[Checkpoint] Saved to {tf12_ckpt}")
    else:
        model12 = StandardTransformer(vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=12, max_seq_len=256)
        model12.load_state_dict(torch.load(tf12_ckpt, map_location=device))
        print(f"[OK] Loaded {tf12_ckpt}")

    model12.to(device)
    res_tf12 = evaluate(model12, val, device)
    tp_tf12 = sum(p.numel() for p in model12.parameters())
    print(f"PPL={res_tf12['ppl']:.2f}, Acc={res_tf12['acc']:.4f}, Params={tp_tf12:,}")
    results['TF-12L'] = {'ppl': res_tf12['ppl'], 'acc': res_tf12['acc'], 'params': tp_tf12}

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  FAIR COMPARISON RESULTS")
    print(f"{'='*100}")
    print(f"\nPPoT Architecture:")
    print(f"  Total params: {tp_ppo:,} (4 experts × 3 layers + 3 PP)")
    print(f"  Active params per forward: ~{tp_ppo//2 + sum(p.numel() for p in ppo_model.post_processing.transformer.parameters()):,} (2 experts × 3 layers + 3 PP)")
    print(f"  Effective compute: 2×3 + 3 = 9 layers")
    print()
    print(f"{'Model':<20} {'PPL':>8} {'Acc':>8} {'Params':>10} {'Type':<15}")
    print("-"*100)
    print(f"{'PPoT':<20} {results['PPoT']['ppl']:>8.2f} {results['PPoT']['acc']:>8.4f} {results['PPoT']['params']:>10,} {'MoE (sparse)':<15}")
    print(f"{'Transformer-6L':<20} {results['TF-6L']['ppl']:>8.2f} {results['TF-6L']['acc']:>8.4f} {results['TF-6L']['params']:>10,} {'Dense':<15}")
    print(f"{'Transformer-12L':<20} {results['TF-12L']['ppl']:>8.2f} {results['TF-12L']['acc']:>8.4f} {results['TF-12L']['params']:>10,} {'Dense':<15}")

    print(f"\n{'='*70}")
    print(f"  VS Transformer-6L (active params match)")
    print(f"{'='*70}")
    ppl_improve = (results['TF-6L']['ppl'] - results['PPoT']['ppl']) / results['TF-6L']['ppl'] * 100
    acc_improve = (results['PPoT']['acc'] - results['TF-6L']['acc']) / results['TF-6L']['acc'] * 100
    print(f"  PPL improvement: {ppl_improve:+.1f}%")
    print(f"  Acc improvement: {acc_improve:+.1f}%")

    print(f"\n{'='*70}")
    print(f"  VS Transformer-12L (total params match)")
    print(f"{'='*70}")
    ppl_improve = (results['TF-12L']['ppl'] - results['PPoT']['ppl']) / results['TF-12L']['ppl'] * 100
    acc_improve = (results['PPoT']['acc'] - results['TF-12L']['acc']) / results['TF-12L']['acc'] * 100
    print(f"  PPL improvement: {ppl_improve:+.1f}%")
    print(f"  Acc improvement: {acc_improve:+.1f}%")


if __name__ == "__main__":
    main()
