"""
Load saved checkpoints and evaluate PPoT vs Transformer on wikitext-103.
Outputs ALL meaningful metrics for paper writing.

Usage:
    python3 evaluate_checkpoints.py --dataset wikitext-103

"""
import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer


def load_wikitext(cache_dir, dataset="wikitext-2", num_samples=50000):
    candidate_names = []
    if dataset.startswith("wikitext-"):
        candidate_names.append(dataset)
        candidate_names.append("wikitext-" + dataset)
    else:
        candidate_names.append("wikitext-" + dataset)
    candidate_names += ["wikitext-103-raw", "wikitext-2-raw", "wikitext-103", "wikitext-2"]
    seen = set()
    unique = []
    for n in candidate_names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    for name in unique:
        for suffix in ["raw", "tokens"]:
            path = os.path.join(cache_dir, name, f"wiki.train.{suffix}")
            if os.path.exists(path):
                print(f"[Data] Using: {path}")
                if suffix == "raw":
                    with open(path, "r") as f:
                        lines = [s.strip() for s in f if len(s.strip()) > 20]
                else:
                    with open(path, "r") as f:
                        text = f.read()
                    lines = [s.strip() for s in text.split("\n") if len(s.strip()) > 20]
                lines = lines[:num_samples]
                print(f"[Data] Loaded {len(lines)} lines")
                return lines

    raise FileNotFoundError(f"Wikitext dataset not found! Dataset: {dataset}")


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def evaluate(model, dataloader, device, model_type="poe"):
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
    expert_outs = torch.stack(expert_outs, dim=0)
    mean_out = expert_outs.mean(dim=[1, 2])
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


class StandardTransformer(torch.nn.Module):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-103", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    cache_dir = "."
    texts = load_wikitext(cache_dir, args.dataset, args.samples)

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(texts, tokenizer, 256)
    tr, va = torch.utils.data.random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val, {len(trl)} batches/train")

    d = args.d_model

    # ============================================================
    # Load and evaluate Transformer
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Transformer-5L")
    print(f"{'='*70}")

    tf_model = StandardTransformer(vocab_size=50257, d_model=d, n_head=4, d_ff=d*2, num_layers=5, max_seq_len=256)
    tf_checkpoint = "model_checkpoints/transformer.pt"
    if os.path.exists(tf_checkpoint):
        tf_model.load_state_dict(torch.load(tf_checkpoint, map_location=device))
        print(f"[Checkpoint] Loaded from {tf_checkpoint}")
    else:
        print(f"[Error] Checkpoint not found: {tf_checkpoint}")
        return

    tf_model.to(device)
    res_tf = evaluate(tf_model, val, device)
    tp_tf = sum(p.numel() for p in tf_model.parameters())
    print(f"PPL={res_tf['ppl']:.2f}, Acc={res_tf['acc']:.4f}, Params={tp_tf:,}")

    # ============================================================
    # Load and evaluate PPoT
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PPoT")
    print(f"{'='*70}")

    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=3,
                    d_model=d, n_head=4, d_ff=d*2, top_k=2, max_seq_len=256,
                    batch_size=16, num_epochs=5, learning_rate=3e-4,
                    lb_loss_weight=0.1, div_loss_weight=0.5, num_gpus=2)

    ppo_model = PoEModel(cfg)
    ppo_checkpoint = "model_checkpoints/ppot.pt"
    if os.path.exists(ppo_checkpoint):
        ppo_model.load_state_dict(torch.load(ppo_checkpoint, map_location=device))
        print(f"[Checkpoint] Loaded from {ppo_checkpoint}")
    else:
        print(f"[Error] Checkpoint not found: {ppo_checkpoint}")
        return

    ppo_model.to(device)
    res_ppo = evaluate(ppo_model, val, device)
    stats = routing_stats(ppo_model, trl, device)
    div = expert_diversity_check(ppo_model, val, device)
    tp_ppo = sum(p.numel() for p in ppo_model.parameters())

    print(f"PPL={res_ppo['ppl']:.2f}, Acc={res_ppo['acc']:.4f}, Params={tp_ppo:,}")
    print(f"Balance={stats['balance']:.3f}")
    print(f"Experts: {stats['counts']}")
    print(f"VN Entropy: {div['normalized']*100:.1f}%")

    # ============================================================
    # Summary Table
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  COMPREHENSIVE RESULTS (Loaded from Checkpoints)")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'PPL':>8} {'Acc':>8} {'Params':>10} {'Balance':>8} {'VN%':>6} {'Eff':>5} {'Time':>8}")
    print("-"*90)
    print(f"{'Transformer-5L':<20} {res_tf['ppl']:>8.2f} {res_tf['acc']:>8.4f} {tp_tf:>10,} {'—':>8} {'—':>6} {'—':>5} {'—':>8}")
    print(f"{'PPoT':<20} {res_ppo['ppl']:>8.2f} {res_ppo['acc']:>8.4f} {tp_ppo:>10,} {stats['balance']:>8.3f} {div['normalized']*100:>5.1f}% {4.00:>5.2f} {'—':>8}")

    # ============================================================
    # Comparison
    # ============================================================
    ppl_improve = (res_tf["ppl"] - res_ppo["ppl"]) / res_tf["ppl"] * 100
    acc_improve = (res_ppo["acc"] - res_tf["acc"]) / res_tf["acc"] * 100
    print(f"\n{'='*70}")
    print(f"  PPoT vs Transformer-5L")
    print(f"{'='*70}")
    print(f"  PPL improvement: {ppl_improve:+.1f}%")
    print(f"  Acc improvement: {acc_improve:+.1f}%")
    print(f"\n  Transformer-5L: PPL={res_tf['ppl']:.2f}, Acc={res_tf['acc']:.4f}")
    print(f"  PPoT:           PPL={res_ppo['ppl']:.2f}, Acc={res_ppo['acc']:.4f}")


if __name__ == "__main__":
    main()
