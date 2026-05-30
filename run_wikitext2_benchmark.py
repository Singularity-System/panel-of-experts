"""
Comprehensive PPoT vs Transformer Benchmark on Wikitext-2.
Outputs ALL meaningful metrics for paper writing.

Usage:
    # On AutoDL (has internet access):
    bash run_wikitext2_benchmark.sh

    # Or directly:
    python3 run_wikitext2_benchmark.py --samples 50000 --epochs 5 --d_model 128

Downloads wikitext-2 automatically, trains all configs, outputs comprehensive table.
"""
import argparse
import os
import urllib.request
import zipfile
import io
import torch
import math
import random
import time
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from transformers import GPT2Model, GPT2Config, get_linear_schedule_with_warmup
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer


# ============================================================
# Data Loading
# ============================================================

def download_wikitext2(cache_dir="."):
    """Download wikitext-2 from multiple fallback sources, or detect existing raw format."""
    # Check for raw format (one sentence per line)
    for ds in ["wikitext-103-raw", "wikitext-2-raw", "wikitext-2", "wikitext"]:
        raw_path = os.path.join(cache_dir, ds, "wiki.train.raw")
        if os.path.exists(raw_path):
            print(f"[Data] Found raw format at {raw_path}")
            return cache_dir
        token_path = os.path.join(cache_dir, ds, "wiki.train.tokens")
        if os.path.exists(token_path):
            print(f"[Data] Found tokens format at {token_path}")
            return cache_dir

    urls = [
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v10.zip",
        "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-v1.zip",
    ]

    for url in urls:
        print(f"[Data] Downloading from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for name in zf.namelist():
                    if "wiki.train.tokens" in name or "wiki.train.raw" in name:
                        zf.extract(name, cache_dir)
                        print(f"[Data] Extracted: {name}")
                        return cache_dir
        except Exception as e:
            print(f"[Data] Failed: {e}")
            continue

    raise RuntimeError("All wikitext-2 download sources failed!")


def load_wikitext(cache_dir, dataset="wikitext-2", num_samples=50000):
    """Load wikitext dataset (supports raw and tokens formats)."""
    # Try multiple locations
    search_paths = [
        os.path.join(cache_dir, f"wikitext-{dataset}-raw", "wiki.train.raw"),
        os.path.join(cache_dir, f"wikitext-{dataset}", "wiki.train.raw"),
        os.path.join(cache_dir, f"wikitext-{dataset}", "wiki.train.tokens"),
        os.path.join(cache_dir, "wikitext-2-raw", "wiki.train.raw"),
        os.path.join(cache_dir, "wikitext-2", "wiki.train.raw"),
        os.path.join(cache_dir, "wikitext-2", "wiki.train.tokens"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"[Data] Using: {path}")
            if path.endswith(".raw"):
                with open(path, "r") as f:
                    lines = [s.strip() for s in f if len(s.strip()) > 20]
            else:
                with open(path, "r") as f:
                    text = f.read()
                lines = [s.strip() for s in text.split("\n") if len(s.strip()) > 20]
            lines = lines[:num_samples]
            print(f"[Data] Loaded {len(lines)} lines")
            return lines

    raise FileNotFoundError(f"Wikitext not found! Searched: {search_paths}")


# ============================================================
# Collation & Dataset
# ============================================================

def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


# ============================================================
# Standard Transformer
# ============================================================

class StandardTransformer(torch.nn.Module):
    """Standard Transformer from scratch — accurate param counting."""
    def __init__(self, vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=8, max_seq_len=256):
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


# ============================================================
# Training
# ============================================================

def train_model(model, trl, epochs, device, seed=0, lb_alpha=0.0, div_alpha=0.0):
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = len(trl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in trl:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            loss = outputs["loss"]
            if lb_alpha > 0 and hasattr(model, "auxiliary_load_balance_loss"):
                loss = loss + lb_alpha * model.auxiliary_load_balance_loss()
            if div_alpha > 0 and hasattr(model, "expert_diversity_loss"):
                loss = loss + div_alpha * model.expert_diversity_loss()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs["loss"].item()
    return model


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            total_loss += outputs["loss"].item()
            logits = outputs["logits"]
            sl = logits[..., :-1, :].contiguous()
            slb = batch["labels"][..., 1:].contiguous()
            sm = batch["attention_mask"][..., 1:].contiguous()
            # Get accuracy per token to avoid OOM on large logits
            preds = sl.argmax(-1)
            total_correct += ((preds == slb.to(preds.device)) & (sm.to(preds.device) == 1)).sum().item()
            total_tokens += sm.sum().item()
    avg = total_loss / len(dataloader)
    return {"loss": avg, "ppl": math.exp(avg), "acc": total_correct / max(total_tokens, 1)}


# ============================================================
# Routing Stats
# ============================================================

def routing_stats(model, dataloader, device, num_batches=50):
    hook = {"w": None, "i": None}
    def hook_fn(m, i, o):
        hook["w"] = o[0].detach() if isinstance(o, tuple) else None
        hook["i"] = o[1].detach() if isinstance(o, tuple) and len(o) > 1 else None
    h = model.router.register_forward_hook(hook_fn)
    counts = torch.zeros(model.num_experts, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            if hook["i"] is None:
                continue
            B, S, _ = hook["i"].shape
            for k in range(model.top_k):
                flat = hook["i"][:, :, k].reshape(-1)
                mask = batch["attention_mask"].reshape(-1)
                valid = flat[mask.bool()]
                for e in range(model.num_experts):
                    counts[e] += (valid == e).sum().item()
    h.remove()
    total = counts.sum().item()
    balance = counts.min().item() / max(counts.max().item(), 1)
    util = counts / max(total, 1)
    return {"counts": counts.tolist(), "balance": balance, "utilization": util.tolist(), "total": total}


# ============================================================
# Expert Diversity (Von Neumann Entropy)
# ============================================================

def expert_diversity_check(model, dataloader, device):
    model.eval()
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"][:8].to(device)
    attention_mask = batch["attention_mask"][:8].to(device)
    B, S = input_ids.shape
    pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    x = model.wte(input_ids) + model.wpe(pos_ids)
    expert_outs = []
    for e in range(model.num_experts):
        dev = model.expert_devices[e]
        out = model.experts[e](x.to(dev), attention_mask.to(dev))
        expert_outs.append(out.cpu())
    expert_outs = torch.stack(expert_outs, dim=0)  # (E, B, S, D)
    mean_out = expert_outs.mean(dim=[1, 2])  # (E, D)
    norms = mean_out.norm(dim=-1, keepdim=True)
    normalized = mean_out / norms
    gram = normalized @ normalized.T
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, min=1e-8)
    p = eigvals / eigvals.sum()
    entropy = -(p * p.log()).sum()
    max_ent = math.log(model.num_experts)
    return {
        "von_neumann_entropy": entropy.item(),
        "max_entropy": max_ent,
        "normalized": entropy.item() / max_ent,
        "eigvals": eigvals.tolist(),
    }


# ============================================================
# Router Collapse Diagnostic
# ============================================================

def check_router_collapse(stats, num_experts):
    counts = stats["counts"]
    total = sum(counts)
    max_pct = max(counts) / total * 100
    min_pct = min(counts) / total * 100
    dead_experts = sum(1 for c in counts if c < total * 0.01)
    return {
        "max_expert_pct": max_pct,
        "min_expert_pct": min_pct,
        "dead_experts": dead_experts,
        "effective_experts": total / (sum(c**2 for c in counts) / max(total, 1)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-2", choices=["wikitext-2", "wikitext-103"],
                       help="Dataset to use")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    cache_dir = download_wikitext2()
    texts = load_wikitext(cache_dir, args.dataset, args.samples)

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(texts, tokenizer, 256)
    tr, va = random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val, {len(trl)} batches/train")

    d = args.d_model
    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
                    d_model=d, n_head=4, d_ff=d*2, top_k=2, max_seq_len=256,
                    batch_size=16, num_epochs=args.epochs, learning_rate=3e-4,
                    lb_loss_weight=0.1, div_loss_weight=0.5)

    # ============================================================
    # Experiment Configs
    # ============================================================
    configs = [
        # (name, type, kwargs)
        ("PPoT-Baseline", "poe", {"lb": 0.0, "div": 0.0}),
        ("PPoT-LB", "poe", {"lb": 0.1, "div": 0.0}),
        ("PPoT-Div", "poe", {"lb": 0.0, "div": 0.5}),
        ("PPoT-LB+Div", "poe", {"lb": 0.1, "div": 0.5}),
        ("Transformer-5L", "tf", {"nl": 5}),
        ("Transformer-8L", "tf", {"nl": 8}),
        ("Transformer-12L", "tf", {"nl": 12}),
    ]

    results = []
    for name, mtype, kwargs in configs:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        t0 = time.time()

        if mtype == "poe":
            model = PoEModel(cfg)
            train_model(model, trl, args.epochs, device, args.seed,
                       lb_alpha=kwargs["lb"], div_alpha=kwargs["div"])
            res = evaluate(model, val, device)
            stats = routing_stats(model, trl, device)
            collapse = check_router_collapse(stats, cfg.num_experts)
            div = expert_diversity_check(model, val, device)

            tp = sum(p.numel() for p in model.parameters())
            embed_p = sum(p.numel() for n, p in model.named_parameters() if 'wte' in n or 'wpe' in n)
            print(f"PPL={res['ppl']:.2f}, Acc={res['acc']:.4f}, Balance={stats['balance']:.3f}")
            print(f"Experts: {stats['counts']}")
            print(f"VN Entropy: {div['normalized']*100:.1f}%")
            print(f"Dead experts: {collapse['dead_experts']}, Effective: {collapse['effective_experts']:.2f}")

            results.append({
                "name": name, "type": "PPoT",
                "ppl": res["ppl"], "acc": res["acc"], "loss": res["loss"],
                "params": tp, "embed_params": embed_p,
                "balance": stats["balance"], "counts": stats["counts"],
                "vn_entropy": div["von_neumann_entropy"],
                "vn_max": div["max_entropy"],
                "vn_normalized": div["normalized"],
                "eigvals": div["eigvals"],
                "dead_experts": collapse["dead_experts"],
                "effective_experts": collapse["effective_experts"],
                "time": time.time() - t0,
            })
        else:
            nl = kwargs["nl"]
            model = StandardTransformer(vocab_size=50257, d_model=d, n_head=4, d_ff=d*2, num_layers=nl, max_seq_len=256)
            train_model(model, trl, args.epochs, device, args.seed)
            res = evaluate(model, val, device)
            tp = sum(p.numel() for p in model.parameters())
            embed_p = sum(p.numel() for n, p in model.named_parameters() if 'wte' in n or 'wpe' in n)
            print(f"PPL={res['ppl']:.2f}, Acc={res['acc']:.4f}, Params={tp:,}")

            results.append({
                "name": name, "type": "Transformer",
                "ppl": res["ppl"], "acc": res["acc"], "loss": res["loss"],
                "params": tp, "embed_params": embed_p,
                "time": time.time() - t0,
            })

    # ============================================================
    # Summary Table
    # ============================================================
    print(f"\n{'='*110}")
    print(f"  COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*110}")
    print(f"{'Model':<20} {'Type':<12} {'PPL':>8} {'Acc':>8} {'Params':>10} {'Balance':>8} {'VN%':>6} {'Dead':>4} {'Eff':>5} {'Time':>8}")
    print("-"*110)
    for r in results:
        if r["type"] == "PPoT":
            print(f"{r['name']:<20} {r['type']:<12} {r['ppl']:>8.2f} {r['acc']:>8.4f} {r['params']:>10,} {r['balance']:>8.3f} {r['vn_normalized']*100:>5.1f}% {r['dead_experts']:>4} {r['effective_experts']:>5.2f} {r['time']:>7.1f}s")
        else:
            print(f"{r['name']:<20} {r['type']:<12} {r['ppl']:>8.2f} {r['acc']:>8.4f} {r['params']:>10,} {'—':>8} {'—':>6} {'—':>4} {'—':>5} {r['time']:>7.1f}s")

    # ============================================================
    # Comparisons vs Baseline
    # ============================================================
    baseline = [r for r in results if r["name"] == "PPoT-Baseline"][0]
    print(f"\n{'='*70}")
    print(f"  VS PPoT-Baseline")
    print(f"{'='*70}")
    for r in results:
        if r["name"] == "PPoT-Baseline": continue
        ppl_d = (r["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100
        acc_d = (r["acc"] - baseline["acc"]) / max(baseline["acc"], 1e-8) * 100
        print(f"{r['name']}: PPL {ppl_d:+.1f}%, Acc {acc_d:+.1f}%")

    # ============================================================
    # PPoT vs Transformer
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PPoT vs Transformer (same d_model={d})")
    print(f"{'='*70}")
    for r in results:
        if r["type"] == "PPoT":
            ppl_d = (r["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100
            print(f"  {r['name']}: PPL={r['ppl']:.2f}, Acc={r['acc']:.4f} (vs baseline {ppl_d:+.1f}%)")
        else:
            ppl_d_vs_ppo = (r["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100
            print(f"  {r['name']}: PPL={r['ppl']:.2f}, Acc={r['acc']:.4f} (vs PPoT {ppl_d_vs_ppo:+.1f}%)")

    # ============================================================
    # Expert Routing Details
    # ============================================================
    ppo_results = [r for r in results if r["type"] == "PPoT"]
    print(f"\n{'='*70}")
    print(f"  ROUTING DETAILS (PPoT)")
    print(f"{'='*70}")
    for r in ppo_results:
        print(f"  {r['name']}: counts={r['counts']}, balance={r['balance']:.3f}, dead={r['dead_experts']}, eff={r['effective_experts']:.2f}")

    # ============================================================
    # Eigenvalue Details
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  EIGENVALUES (Gram Matrix)")
    print(f"{'='*70}")
    for r in ppo_results:
        eig_str = ", ".join(f"{e:.3f}" for e in r["eigvals"])
        print(f"  {r['name']}: [{eig_str}]")

    # ============================================================
    # Router Collapse Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  ROUTER COLLAPSE ANALYSIS")
    print(f"{'='*70}")
    for r in ppo_results:
        print(f"  {r['name']}: max={r['counts'].index(max(r['counts']))}({max(r['counts'])}), "
              f"min={r['counts'].index(min(r['counts']))}({min(r['counts'])}), "
              f"max_pct={max(r['counts'])/sum(r['counts'])*100:.1f}%")

    # ============================================================
    # Final Recommendation
    # ============================================================
    best_ppo = min([r for r in results if r["type"] == "PPoT"], key=lambda x: x["ppl"])
    best_tf = min([r for r in results if r["type"] == "Transformer"], key=lambda x: x["ppl"])
    print(f"\n{'='*70}")
    print(f"  BEST MODELS")
    print(f"{'='*70}")
    print(f"  Best PPoT:     {best_ppo['name']} (PPL={best_ppo['ppl']:.2f}, Acc={best_ppo['acc']:.4f})")
    print(f"  Best TF:       {best_tf['name']} (PPL={best_tf['ppl']:.2f}, Acc={best_tf['acc']:.4f})")
    ppl_improve = (best_tf["ppl"] - best_ppo["ppl"]) / best_tf["ppl"] * 100
    print(f"  PPoT improvement: {ppl_improve:.1f}% PPL reduction")


if __name__ == "__main__":
    main()
