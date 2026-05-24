"""Head-to-head: PoE with vs without LB loss. Same seed, same data."""
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


def train_model(model, trl, epochs, device, lb_alpha=0.0, seed=0):
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
    """Collect routing balance from hard assignments."""
    hook = {"w": None, "i": None}
    def hook_fn(m, i, o):
        hook["w"] = o[0].detach(); hook["i"] = o[1].detach()
    h = model.router.register_forward_hook(hook_fn)
    counts = torch.zeros(model.num_experts, dtype=torch.long)
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            B, S, _ = hook["i"].shape
            total_tokens += B * S
            for k in range(model.top_k):
                flat = hook["i"][:, :, k].reshape(-1)
                mask = attention_mask.reshape(-1)
                valid = flat[mask.bool()]
                for e in range(model.num_experts):
                    counts[e] += (valid == e).sum().item()
    h.remove()
    total = counts.sum().item()
    util = counts / max(total, 1)
    balance = counts.min().item() / max(counts.max().item(), 1)
    return {"counts": counts.tolist(), "balance": balance, "utilization": util.tolist(), "total": total}


def main():
    device = torch.device("cpu")
    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
                    d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=128,
                    batch_size=16, num_epochs=5, learning_rate=3e-4)

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
    print(f"PoE params: {count(PoEModel(cfg).parameters()):,}")
    print(f"Config: {cfg.num_experts} experts, top_k={cfg.top_k}, d_model={cfg.d_model}")

    print("\n" + "="*70)
    print("  WITHOUT LB Loss (α=0)")
    print("="*70)
    model_no_lb = PoEModel(cfg)
    train_model(model_no_lb, trl, epochs=5, device=device, lb_alpha=0.0, seed=42)
    res_no_lb = evaluate(model_no_lb, val, device)
    stats_no_lb = routing_stats(model_no_lb, trl, device, num_batches=50)

    print("\n" + "="*70)
    print("  WITH LB Loss (α=0.1)")
    print("="*70)
    model_with_lb = PoEModel(cfg)
    train_model(model_with_lb, trl, epochs=5, device=device, lb_alpha=0.1, seed=42)
    res_with_lb = evaluate(model_with_lb, val, device)
    stats_with_lb = routing_stats(model_with_lb, trl, device, num_batches=50)

    # Results
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'No LB':>12} {'LB α=0.1':>12} {'Δ':>10}")
    print("-"*56)
    print(f"{'Val Loss':<20} {res_no_lb['loss']:>12.4f} {res_with_lb['loss']:>12.4f} {(res_with_lb['loss']-res_no_lb['loss']):>+.4f}")
    print(f"{'Val PPL':<20} {res_no_lb['ppl']:>12.2f} {res_with_lb['ppl']:>12.2f} {(res_with_lb['ppl']-res_no_lb['ppl']):>+.2f}")
    print(f"{'Val Acc':<20} {res_no_lb['acc']:>12.4f} {res_with_lb['acc']:>12.4f} {(res_with_lb['acc']-res_no_lb['acc']):>+.4f}")
    print(f"{'Balance Ratio':<20} {stats_no_lb['balance']:>12.3f} {stats_with_lb['balance']:>12.3f} {(stats_with_lb['balance']-stats_no_lb['balance']):>+.3f}")

    print(f"\n{'Expert':<10} {'No LB %':>10} {'LB α=0.1 %':>12}")
    print("-"*24)
    for i in range(cfg.num_experts):
        print(f"E{i:<9} {stats_no_lb['utilization'][i]*100:>9.1f}% {stats_with_lb['utilization'][i]*100:>11.1f}%")

    # Conclusion
    ppl_change = (res_with_lb['ppl'] - res_no_lb['ppl']) / res_no_lb['ppl'] * 100
    print(f"\n{'='*70}")
    print("  CONCLUSION")
    print("="*70)
    if ppl_change < 0:
        print(f"LB loss IMPROVES PPL by {abs(ppl_change):.1f}%")
        print(f"Balance: {stats_no_lb['balance']:.3f} → {stats_with_lb['balance']:.3f}")
        print(f"Better router balance → better performance (experts used more evenly)")
    else:
        print(f"LB loss slightly WORSENS PPL by {ppl_change:.1f}%")
        print(f"But balance improves: {stats_no_lb['balance']:.3f} → {stats_with_lb['balance']:.3f}")


if __name__ == "__main__":
    main()
