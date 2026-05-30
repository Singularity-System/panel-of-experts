"""Compare: LB loss vs Diversity loss vs Both. Synthetic data (consistent comparison)."""
import os
import torch
import math
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer


def count(p):
    return sum(x.numel() for x in p)


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def generate_synthetic_data(num_samples=50000, seed=42):
    """Generate synthetic text data with patterns that benefit from expert specialization."""
    import random
    rng = random.Random(seed)
    texts = []
    for i in range(num_samples):
        pat = i % 3
        length = 80 + rng.randint(0, 40)
        if pat == 0:
            # Pattern 1: repetitive low-range tokens
            tokens = [rng.randint(100, 200) for _ in range(length)]
        elif pat == 1:
            # Pattern 2: alternating medium-range tokens
            tokens = [rng.randint(200, 300) + (i % 3) * 50 for i in range(length)]
        else:
            # Pattern 3: structured high-range tokens
            tokens = [rng.randint(300, 500) if j % 5 != 0 else 100 + (j % 4) * 100 for j in range(length)]
        texts.append(" ".join(str(t) for t in tokens))
    return texts


def train_model(model, trl, epochs, device, lb_alpha=0.0, div_alpha=0.0, seed=0):
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
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
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
            sl = logits[..., :-1, :].contiguous(); slb = batch["labels"][..., 1:].contiguous()
            sm = batch["attention_mask"][..., 1:].contiguous()
            total_correct += ((sl.argmax(-1) == slb) & (sm == 1)).sum().item()
            total_tokens += sm.sum().item()
    avg = total_loss / len(dataloader)
    return {"loss": avg, "ppl": math.exp(avg), "acc": total_correct / max(total_tokens, 1)}


def routing_stats(model, dataloader, device, num_batches=30):
    hook = {"w": None, "i": None}
    def hook_fn(m, i, o):
        hook["w"] = o[0].detach(); hook["i"] = o[1].detach()
    h = model.router.register_forward_hook(hook_fn)
    counts = torch.zeros(model.num_experts, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            B, S, _ = hook["i"].shape
            for k in range(model.top_k):
                flat = hook["i"][:, :, k].reshape(-1)
                mask = batch["attention_mask"].reshape(-1)
                valid = flat[mask.bool()]
                for e in range(model.num_experts):
                    counts[e] += (valid == e).sum().item()
    h.remove()
    total = counts.sum().item()
    util = counts / max(total, 1)
    balance = counts.min().item() / max(counts.max().item(), 1)
    return {"counts": counts.tolist(), "balance": balance, "utilization": util.tolist(), "total": total}


def expert_diversity_check(model, dataloader, device):
    """Check actual expert diversity via Gram matrix."""
    model.eval()
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"][:4].to(device)
    attention_mask = batch["attention_mask"][:4].to(device)
    B, S = input_ids.shape

    pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    x = model.wte(input_ids) + model.wpe(pos_ids)

    expert_outs = []
    for e in range(model.num_experts):
        dev = model.expert_devices[e]
        out = model.experts[e](x.to(dev), attention_mask.to(dev))
        expert_outs.append(out.cpu())
    expert_outs = torch.stack(expert_outs, dim=0)  # (E, B, S, D)

    # Mean expert output (E, D)
    mean_out = expert_outs.mean(dim=[1, 2])  # (E, D)
    norms = mean_out.norm(dim=-1, keepdim=True)
    normalized = mean_out / norms
    gram = normalized @ normalized.T
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, min=1e-8)
    p = eigvals / eigvals.sum()
    von_neumann = -(p * p.log()).sum()
    max_ent = math.log(model.num_experts)

    return {
        "von_neumann_entropy": von_neumann.item(),
        "max_entropy": max_ent,
        "normalized": von_neumann.item() / max_ent,
        "eigvals": eigvals.tolist(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate synthetic data (default 5000 for quick testing)
    num_samples = int(os.environ.get("NUM_SAMPLES", "5000"))
    texts = generate_synthetic_data(num_samples)
    print(f"Synthetic data: {len(texts)} samples")

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)
    ds = DS(texts, tokenizer, 128)
    tr, va = random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val")

    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
                    d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=128,
                    batch_size=16, num_epochs=3, learning_rate=3e-4)
    print(f"PoE params: {count(PoEModel(cfg).parameters()):,}")

    conditions = [
        (0.0, 0.0, "Baseline"),
        (0.1, 0.0, "LB α=0.1"),
        (0.0, 0.5, "Div α=0.5"),
        (0.1, 0.5, "LB+Div"),
    ]

    results = []
    for lb_a, div_a, label in conditions:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        model = PoEModel(cfg)
        train_model(model, trl, epochs=3, device=device, lb_alpha=lb_a, div_alpha=div_a, seed=42)
        res = evaluate(model, val, device)
        stats = routing_stats(model, trl, device, num_batches=30)
        div_info = expert_diversity_check(model, val, device)

        results.append({"label": label, "res": res, "stats": stats, "div": div_info})
        print(f"PPL={res['ppl']:.2f}, Acc={res['acc']:.4f}, Balance={stats['balance']:.3f}")
        print(f"Von Neumann: {div_info['von_neumann_entropy']:.3f}/{div_info['max_entropy']:.3f} ({div_info['normalized']*100:.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"{'Condition':<15} {'PPL':>8} {'Acc':>8} {'Balance':>8} {'VN Entropy':>12} {'VN Norm':>8}")
    print("-"*63)
    for r in results:
        print(f"{r['label']:<15} {r['res']['ppl']:>8.2f} {r['res']['acc']:>8.4f} {r['stats']['balance']:>8.3f} {r['div']['von_neumann_entropy']:>12.3f} {r['div']['normalized']*100:>7.1f}%")

    baseline = results[0]
    print(f"\n{'='*70}")
    print("  COMPARISON vs Baseline")
    print("="*70)
    for r in results[1:]:
        ppl_delta = (r['res']['ppl'] - baseline['res']['ppl']) / baseline['res']['ppl'] * 100
        bal_delta = (r['stats']['balance'] - baseline['stats']['balance']) / max(baseline['stats']['balance'], 0.001) * 100
        vn_delta = r['div']['normalized'] - baseline['div']['normalized']
        print(f"{r['label']}: PPL {ppl_delta:+.1f}%, Balance {baseline['stats']['balance']:.3f}→{r['stats']['balance']:.3f} ({bal_delta:+.0f}%), VN {baseline['div']['normalized']*100:.1f}%→{r['div']['normalized']*100:.1f}% ({vn_delta*100:+.1f}%)")

    # Gram matrix details
    print(f"\n{'='*70}")
    print("  GRAM MATRIX EIGENVALUES")
    print("="*70)
    for r in results:
        print(f"{r['label']}: eigvals={[f'{e:.3f}' for e in r['div']['eigvals']]}")


if __name__ == "__main__":
    main()
