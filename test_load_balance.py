"""Add load balancing loss to PoE router and run comparison experiment."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
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


class RouterWithLB(nn.Module):
    """Router that also computes load balancing auxiliary loss (Switch Transformer style)."""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_linear = nn.Linear(d_model, num_experts)
        self._last_full_probs = None  # (B, S, num_experts) softmax probs
        self._last_indices = None     # (B, S, top_k)

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate_linear(hidden_states)  # (B, S, num_experts)
        full_probs = torch.softmax(logits, dim=-1)
        self._last_full_probs = full_probs  # NOT detached - needed for gradient flow
        self._last_full_probs_detached = full_probs.detach()  # for stats

        if self.top_k >= self.num_experts:
            indices = torch.arange(self.num_experts, device=hidden_states.device).unsqueeze(0).unsqueeze(0).expand(full_probs.shape[0], full_probs.shape[1], -1)
            top_weights = full_probs
        else:
            top_weights, top_indices = torch.topk(full_probs, self.top_k, dim=-1)
            top_weights = torch.nn.functional.normalize(top_weights, p=1, dim=-1)
            indices = top_indices
        self._last_indices = indices.detach()
        return top_weights, indices

    def load_balance_loss(self):
        """Switch Transformer: α * N * Σ(f_i * P_i) where f=hard freq, P=softmax prob."""
        if self._last_full_probs is None:
            return torch.tensor(0.0, device=self.gate_linear.weight.device)

        probs = self._last_full_probs        # (B, S, num_experts) - grad preserved
        indices = self._last_indices         # (B, S, top_k)
        N = self.num_experts

        # P_i = average softmax probability for expert i
        P = probs.mean(dim=[0, 1])  # (num_experts,) — grad flows

        # f_i = hard assignment frequency from top-k indices
        flat_idx = indices.view(-1)  # (B*S*top_k,)
        one_hot = torch.zeros(flat_idx.shape[0], N, device=flat_idx.device)
        one_hot.scatter_(1, flat_idx.unsqueeze(1), 1.0)
        f = one_hot.float().mean(dim=0)  # (num_experts,) — no grad (hard)

        # Loss = N * Σ(f_i * P_i)
        loss = N * (f * P).sum()
        return loss


class PoEModelWithLB(PoEModel):
    """PoE model with router that supports load balancing loss."""
    def __init__(self, config: PoEConfig):
        super().__init__(config)
        # Replace router with LB-capable version
        old_router = self.router
        new_router = RouterWithLB(config.d_model, config.num_experts, config.top_k)
        # Copy weights
        new_router.gate_linear.weight.data.copy_(old_router.gate_linear.weight.data)
        new_router.gate_linear.bias.data.copy_(old_router.gate_linear.bias.data)
        self.router = new_router

    def lb_loss(self):
        return self.router.load_balance_loss()


def collect_routing_stats(model, dataloader, device, num_batches=50, label=""):
    """Collect routing statistics."""
    hook_data = {"weights": None, "indices": None}
    def hook_fn(m, i, o):
        hook_data["weights"] = o[0].detach()
        hook_data["indices"] = o[1].detach()
    handle = model.router.register_forward_hook(hook_fn)
    model.eval()
    expert_counts = torch.zeros(model.num_experts, dtype=torch.long)
    total_tokens = 0
    batch_count = 0

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

    # Full softmax entropy
    full_ent, max_full_ent = 0.0, 0.0
    if hasattr(model.router, '_last_full_probs'):
        hook_data2 = {"full_probs": None}
        def hook_fn2(m, i, o):
            if hasattr(m, '_last_full_probs_detached') and m._last_full_probs_detached is not None:
                hook_data2["full_probs"] = m._last_full_probs_detached
            elif hasattr(m, '_last_full_probs') and m._last_full_probs is not None:
                hook_data2["full_probs"] = m._last_full_probs.detach()
        handle2 = model.router.register_forward_hook(hook_fn2)
        all_full_probs = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches: break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                if hook_data2["full_probs"] is not None:
                    all_full_probs.append(hook_data2["full_probs"].reshape(-1, model.num_experts))
        handle2.remove()
        if all_full_probs:
            flat_fp = torch.cat(all_full_probs)
            full_ent = -(flat_fp * (flat_fp + 1e-10).log()).sum(dim=-1).mean().item()
            max_full_ent = math.log(model.num_experts)
    else:
        all_topk_weights = []
        hook_data3 = {"w": None, "i": None}
        def hook_fn3(m, i, o):
            hook_data3["w"] = o[0].detach()
            hook_data3["i"] = o[1].detach()
        handle3 = model.router.register_forward_hook(hook_fn3)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches: break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                all_topk_weights.append((hook_data3["w"], hook_data3["i"]))
        handle3.remove()
        full_probs_list = []
        for w, idx in all_topk_weights:
            B, S, _ = idx.shape
            fp = torch.zeros(B, S, model.num_experts)
            for k in range(model.top_k):
                for b in range(B):
                    for s in range(S):
                        fp[b, s, idx[b, s, k].item()] = w[b, s, k].item()
            full_probs_list.append(fp.reshape(-1, model.num_experts))
        flat_fp = torch.cat(full_probs_list)
        full_ent = -(flat_fp * (flat_fp + 1e-10).log()).sum(dim=-1).mean().item()
        max_full_ent = math.log(model.num_experts)

    print(f"\n  {label}")
    print(f"  Expert: {', '.join(f'E{int(e)}={util[e]*100:.1f}%' for e in range(model.num_experts))}")
    print(f"  Balance: {balance_ratio:.3f} | Unused: {unused}/{model.num_experts}")
    print(f"  Full softmax entropy: {full_ent:.3f}/{max_full_ent:.3f} ({full_ent/max_full_ent*100:.1f}%)")

    return {"balance": balance_ratio, "entropy": full_ent, "max_entropy": max_full_ent,
            "unused": unused, "utilization": util.tolist(), "counts": expert_counts.tolist()}


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    cfg = PoEConfig(
        num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
        d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=128,
        batch_size=16, num_epochs=5, learning_rate=3e-4,
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

    results = {}

    for lb_alpha, lb_label in [(0.0, "No LB Loss"), (0.01, "LB α=0.01"), (0.1, "LB α=0.1")]:
        print(f"\n{'='*70}")
        print(f"  Condition: {lb_label}")
        print(f"{'='*70}")

        if lb_alpha > 0:
            model = PoEModelWithLB(cfg)
        else:
            model = PoEModel(cfg)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        total_steps = len(trl) * cfg.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            total_loss = 0
            for batch in trl:
                outputs = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device))
                loss = outputs["loss"]
                if lb_alpha > 0 and hasattr(model, 'lb_loss'):
                    lb_loss = model.lb_loss()
                    if isinstance(lb_loss, torch.Tensor) and lb_loss.requires_grad:
                        loss = loss + lb_alpha * lb_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += outputs["loss"].item()

            avg_loss = total_loss / len(trl)
            print(f"  Epoch {epoch} | loss: {avg_loss:.4f}")

        stats_train = collect_routing_stats(model, trl, device, num_batches=50,
                                            label=f"After Training ({lb_label})")
        stats_val = collect_routing_stats(model, val, device, num_batches=50,
                                          label=f"Validation ({lb_label})")
        results[lb_label] = stats_train

    # === Summary ===
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'Balance':>10} {'Entropy':>10} {'Unused':>8} {'Top Expert':>10}")
    print("-"*70)
    for label, stats in results.items():
        top = max(stats["counts"])
        total = sum(stats["counts"])
        print(f"{label:<20} {stats['balance']:>10.3f} {stats['entropy']:>10.3f} {stats['unused']:>8} {top/total*100:>9.1f}%")

    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print(f"{'='*70}")
    no_lb = results["No LB Loss"]
    best_label = None
    best_balance = 0
    for label in ["LB α=0.01", "LB α=0.1"]:
        if label in results and results[label]["balance"] > best_balance:
            best_balance = results[label]["balance"]
            best_label = label

    if best_label:
        improvement = best_balance / max(no_lb["balance"], 0.001)
        print(f"LB loss improves balance by {improvement:.1f}x ({best_label})")
        if best_balance > 0.3:
            print(f"→ LB loss IS effective for this PoE configuration")
        else:
            print(f"→ LB loss NOT sufficient, consider architecture change")


if __name__ == "__main__":
    main()
