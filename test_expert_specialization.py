"""Check expert specialization: which pattern does each expert prefer?"""
import torch
import random
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from alpha_eai.config import PoEConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def generate_data(num_samples=2000, seed=42):
    """Each sample tagged with its pattern."""
    rng = random.Random(seed)
    data = []
    for i in range(num_samples):
        pat = i % 3
        length = 80 + rng.randint(0, 40)
        if pat == 0:
            tokens = [rng.randint(100, 200) for _ in range(length)]
        elif pat == 1:
            tokens = [rng.randint(200, 300) + (i % 3) * 50 for i in range(length)]
        else:
            tokens = [rng.randint(300, 500) if j % 5 != 0 else 100 + (j % 4) * 100 for j in range(length)]
        data.append((" ".join(str(t) for t in tokens), pat))
    return data


def train_model(model, trl, epochs, device, lb_alpha=0.0, div_alpha=0.0, seed=0):
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = len(trl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)
    for epoch in range(1, epochs + 1):
        model.train()
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


def main():
    device = torch.device("cpu")

    # Check if already trained
    model_path = "ppo_specialization.pt"
    already_trained = __import__('os').path.exists(model_path)

    data = generate_data(2000)
    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(texts, tokenizer, 128)
    tr, va = random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    # Use same seed for reproducibility
    tr_idx = tr.indices
    va_idx = va.indices

    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
                    d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=128,
                    batch_size=16, num_epochs=5, learning_rate=3e-4,
                    lb_loss_weight=0.1, div_loss_weight=0.5)

    if already_trained:
        print("=== Loading pre-trained model ===")
        model = PoEModel(cfg)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("=== Training with LB+Div ===")
        trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
        model = PoEModel(cfg)
        train_model(model, trl, epochs=5, device=device, lb_alpha=0.1, div_alpha=0.5, seed=42)
        torch.save(model.state_dict(), model_path)
        print("Model saved!")

    model.to(device)
    model.eval()

    # Build validation set with correct labels
    val_data = [(texts[i], labels[i]) for i in va_idx]
    val_texts = [t for t, _ in val_data]
    val_labels = [l for _, l in val_data]
    val_ds = DS(val_texts, tokenizer, 128)
    val = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    print(f"\n{'='*70}")
    print(f"  Expert Specialization (LB+Div)")
    print(f"{'='*70}")

    hook = {"i": None}
    def hook_fn(m, i, o):
        hook["i"] = o[1].detach()
    h = model.router.register_forward_hook(hook_fn)

    expert_counts = torch.zeros(3, model.num_experts, dtype=torch.long)

    with torch.no_grad():
        batch_idx = 0
        for batch in val:
            B = batch["input_ids"].shape[0]
            model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            for k in range(model.top_k):
                indices = hook["i"][:, :, k]  # (B, S)
                for b in range(B):
                    global_idx = va_idx[batch_idx * 16 + b] if batch_idx * 16 + b < len(va_idx) else None
                    if global_idx is None:
                        continue
                    pattern_idx = labels[global_idx]
                    expert_indices = indices[b].reshape(-1)
                    valid_mask = batch["attention_mask"][b].reshape(-1).bool()
                    valid_indices = expert_indices[valid_mask]
                    for e in range(model.num_experts):
                        expert_counts[pattern_idx, e] += (valid_indices == e).sum().item()
            batch_idx += 1
    h.remove()

    for p in range(3):
        total = expert_counts[p].sum().item()
        pct = [f"{expert_counts[p, e].item() / max(total, 1) * 100:.1f}%" for e in range(model.num_experts)]
        best_e = expert_counts[p].argmax().item()
        worst_e = expert_counts[p].argmin().item()
        print(f"  Pattern {p}: E0={pct[0]} E1={pct[1]} E2={pct[2]} E3={pct[3]}  → prefers E{best_e}, avoids E{worst_e}")

    overall_balance = expert_counts.min().item() / max(expert_counts.max().item(), 1)
    print(f"\n  Overall balance: {overall_balance:.3f}")

    # Chi-square test: is routing pattern-dependent?
    print(f"\n{'='*70}")
    print(f"  Chi-square test (is expert selection pattern-dependent?)")
    print(f"{'='*70}")
    for e in range(model.num_experts):
        col = expert_counts[:, e].float()
        total_e = col.sum().item()
        expected = torch.tensor([col.sum() / 3 for _ in range(3)])
        chi2 = ((col - expected)**2 / (expected + 1e-8)).sum().item()
        print(f"  Expert {e}: counts={[expert_counts[p, e].item() for p in range(3)]}, chi2={chi2:.2f}")


if __name__ == "__main__":
    main()
