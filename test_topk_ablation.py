"""Ablation: effect of top-k on PoE performance and routing."""
import torch
import math
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
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
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def train_model(model, trl, epochs, device, lb_alpha=0.1, seed=0):
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
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            B, S, _ = hook["i"].shape
            for k in range(model.top_k):
                flat = hook["i"][:, :, k].reshape(-1)
                mask = batch["attention_mask"][..., 1:].reshape(-1) if batch["attention_mask"].shape[-1] > 1 else batch["attention_mask"].reshape(-1)
                valid = flat[mask.bool()] if mask.numel() == flat.numel() else flat
                for e in range(model.num_experts):
                    counts[e] += (valid == e).sum().item()
    h.remove()
    total = counts.sum().item()
    util = counts / max(total, 1)
    balance = counts.min().item() / max(counts.max().item(), 1)
    return {"counts": counts.tolist(), "balance": balance, "utilization": util.tolist(), "total": total}


def main():
    device = torch.device("cpu")
    base_cfg = dict(num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
                    d_model=128, n_head=4, d_ff=256, max_seq_len=128,
                    batch_size=16, num_epochs=5, learning_rate=3e-4, lb_loss_weight=0.1)

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    texts = make_demo_data() * 10
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

    top_k_values = [1, 2, 3, 4]
    results = {}

    for top_k in top_k_values:
        cfg = PoEConfig(**base_cfg, top_k=top_k)
        print(f"\n{'='*60}")
        print(f"  top_k = {top_k}")
        print(f"{'='*60}")

        model = PoEModel(cfg)
        active_params = count(model.wte.parameters()) + count(model.wpe.parameters()) + count(model.router.parameters()) + count(model.fusion.parameters()) + count(model.post_processing.parameters()) + count(model.lm_head.parameters())
        if top_k > 0:
            active_params += top_k * count(model.experts[0].parameters())
        print(f"Active params (k={top_k}): {active_params:,}")

        train_model(model, trl, epochs=5, device=device, lb_alpha=0.1, seed=42)
        res = evaluate(model, val, device)
        stats = routing_stats(model, trl, device, num_batches=50)
        results[top_k] = {"res": res, "stats": stats, "active_params": active_params}

        util_str = ", ".join(f"E{i}={stats['utilization'][i]*100:.1f}%" for i in range(cfg.num_experts))
        print(f"PPL={res['ppl']:.2f}, Acc={res['acc']:.4f}, Balance={stats['balance']:.3f}")
        print(f"Expert: {util_str}")

    # Summary
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY: top_k")
    print("="*70)
    print(f"{'top_k':<8} {'Active':>12} {'Val PPL':>10} {'Val Acc':>10} {'Balance':>10} {'Active Experts':>15}")
    print("-"*67)
    for top_k in top_k_values:
        r = results[top_k]
        active_experts = sum(1 for u in r["stats"]["utilization"] if u > 0.1)
        print(f"{top_k:<8} {r['active_params']:>12,} {r['res']['ppl']:>10.2f} {r['res']['acc']:>10.4f} {r['stats']['balance']:>10.3f} {active_experts:>3}/4")

    # Analysis
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print("="*70)
    best_k = min(top_k_values, key=lambda k: results[k]["res"]["ppl"])
    best_ppl = results[best_k]["res"]["ppl"]
    print(f"Best top_k: {best_k} (PPL={best_ppl:.2f})")
    for k in top_k_values:
        r = results[k]
        if k != best_k:
            diff = (r["res"]["ppl"] - best_ppl) / best_ppl * 100
            print(f"  top_k={k}: PPL {r['res']['ppl']:.2f} ({diff:+.1f}% vs best)")

    # Active compute cost analysis
    print(f"\n{'='*70}")
    print("  COMPUTE COST (active params)")
    print("="*70)
    params_per_expert = count(PoEConfig(**base_cfg, top_k=1).num_experts)  # placeholder
    for k in top_k_values:
        print(f"  top_k={k}: {results[k]['active_params']:,} active params")


if __name__ == "__main__":
    main()
