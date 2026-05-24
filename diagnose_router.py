"""Diagnose router behavior in PoE: entropy, utilization, collapse."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer
from training.data_demo import make_demo_data
import math


def count(p):
    return sum(x.numel() for x in p)


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(padded),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(padded).clone(),
    }


def collect_routing_stats(model, dataloader, device, num_batches=50, label=""):
    """Collect routing statistics via forward hooks."""
    hook_data = {"weights": None, "indices": None}
    def hook_fn(m, i, o):
        hook_data["weights"] = o[0].detach()
        hook_data["indices"] = o[1].detach()
    handle = model.router.register_forward_hook(hook_fn)
    model.eval()
    expert_counts = torch.zeros(model.num_experts, dtype=torch.long)
    total_tokens = 0
    batch_count = 0
    all_weights_flat = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            B, S, _ = hook_data["indices"].shape
            total_tokens += B * S
            batch_count += 1
            all_weights_flat.append(hook_data["weights"].reshape(-1, model.top_k))

            for k in range(model.top_k):
                flat_idx = hook_data["indices"][:, :, k].reshape(-1)
                flat_mask = attention_mask.reshape(-1)
                valid_idx = flat_idx[flat_mask.bool()]
                for e in range(model.num_experts):
                    expert_counts[e] += (valid_idx == e).sum().item()

    handle.remove()

    total = expert_counts.sum().item()
    util = expert_counts / max(total, 1)
    balance_ratio = expert_counts.min().item() / max(expert_counts.max().item(), 1)
    unused = sum(1 for c in expert_counts if c == 0)
    dominant = expert_counts.argmax().item()

    flat_w = torch.cat(all_weights_flat)
    ent = -(flat_w * (flat_w + 1e-10).log()).sum(dim=-1).mean().item()
    max_ent = math.log(model.top_k)

    print(f"\n{'='*60}")
    print(f"  {label} ({batch_count} batches, ~{total_tokens} tokens)")
    print(f"{'='*60}")
    print(f"Expert utilization: {', '.join(f'E{i}={util[i]*100:.1f}%' for i in range(model.num_experts))}")
    print(f"Expert counts:      {', '.join(f'E{int(e)}={int(expert_counts[e]):>6d}' for e in range(model.num_experts))}")
    print(f"Balance ratio:      {balance_ratio:.3f} (1.0 = perfect)")
    print(f"Top-{model.top_k} entropy: {ent:.3f} / {max_ent:.3f} ({ent/max_ent*100:.1f}%)")
    if unused > 0:
        print(f"WARNING: {unused}/{model.num_experts} experts NEVER used!")
    print(f"Dominant expert:    E{dominant} ({expert_counts[dominant]/max(total,1)*100:.1f}% routes)")

    return {
        "utilization": util.tolist(), "entropy": ent, "max_entropy": max_ent,
        "balance_ratio": balance_ratio, "expert_counts": expert_counts.tolist(),
        "unused": unused, "total": total,
    }


def expert_diversity(model, val_loader, device):
    """Check if experts produce different outputs for the same input."""
    print("\n" + "="*60)
    print("  Expert Output Diversity")
    print("="*60)
    model.eval()
    batch = next(iter(val_loader))
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
    expert_outs = torch.stack(expert_outs, dim=0)

    print(f"Pairwise cosine similarity (mean over tokens, B={B}, S={S}):")
    sims = []
    for e1 in range(model.num_experts):
        row = []
        for e2 in range(e1+1, model.num_experts):
            o1 = expert_outs[e1].mean(dim=1)
            o2 = expert_outs[e2].mean(dim=1)
            sim = torch.nn.functional.cosine_similarity(o1, o2, dim=-1).mean().item()
            row.append(f"E{e1}-E{e2}: {sim:.4f}")
            sims.append(sim)
        print(f"  {'  '.join(row)}")

    std_across = expert_outs.std(dim=0).mean().item()
    print(f"Std across experts: {std_across:.4f}")
    return std_across, sims


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    cfg = PoEConfig(
        num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
        d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=128,
        batch_size=16, num_epochs=3, learning_rate=3e-4,
    )

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

    # === EXPERIMENT: Router diagnostics ===
    print("\n" + "="*70)
    print("  Router Behavior Diagnosis")
    print("="*70)
    model = PoEModel(cfg)
    model.to(device)
    print(f"PoE params: {count(model.parameters()):,}")

    stats_before = collect_routing_stats(model, trl, device, num_batches=10, label="Random Router (before training)")

    print("\n--- Training ---")
    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(trl) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in trl:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            loss = outputs["loss"]
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | loss: {total_loss/len(trl):.4f}")

    stats_train = collect_routing_stats(model, trl, device, num_batches=50, label="Trained Router (train set)")
    stats_val = collect_routing_stats(model, val, device, num_batches=50, label="Trained Router (val set)")
    diversity, sims = expert_diversity(model, val, device)

    # === Save summary ===
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"Random router entropy:  {stats_before['entropy']:.3f}/{stats_before['max_entropy']:.3f} ({stats_before['entropy']/stats_before['max_entropy']*100:.1f}%)")
    print(f"Trained router entropy: {stats_train['entropy']:.3f}/{stats_train['max_entropy']:.3f} ({stats_train['entropy']/stats_train['max_entropy']*100:.1f}%)")
    print(f"Train balance ratio:    {stats_train['balance_ratio']:.3f}")
    print(f"Val balance ratio:      {stats_val['balance_ratio']:.3f}")
    print(f"Unused experts:         {stats_train['unused']}/{cfg.num_experts}")
    print(f"Expert diversity (std): {diversity:.4f}")
    print(f"Pairwise sims:          {', '.join(str(f'{s:.4f}') for s in sims)}")

    # Key diagnosis
    print(f"\n{'='*70}")
    print("  DIAGNOSIS")
    print("="*70)
    entropy_drop = stats_before['entropy'] - stats_train['entropy']
    if entropy_drop > 0.3:
        print(f"- Router entropy dropped by {entropy_drop:.3f} after training → router became MORE selective")
    if stats_train['balance_ratio'] < 0.5:
        print(f"- Balance ratio {stats_train['balance_ratio']:.3f} < 0.5 → ROUTER IMBALANCED, some experts dominate")
    if stats_train['unused'] > 0:
        print(f"- {stats_train['unused']}/{cfg.num_experts} experts NEVER used → ROUTER COLLAPSE DETECTED!")
    if diversity < 0.1:
        print(f"- Expert diversity {diversity:.4f} is LOW → experts produce similar outputs (potential underutilization)")
    if sims:
        avg_sim = sum(sims) / len(sims)
        if avg_sim > 0.9:
            print(f"- Average pairwise similarity {avg_sim:.4f} > 0.9 → experts are highly correlated (not learning diverse features)")


if __name__ == "__main__":
    main()
