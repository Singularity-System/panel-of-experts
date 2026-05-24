"""Final comparison: PoE with vs without LB loss on PPL/loss."""
import torch
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


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()
            logits = outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            correct = ((preds == shift_labels) & (shift_mask == 1)).sum().item()
            tokens = shift_mask.sum().item()
            total_correct += correct
            total_tokens += tokens
    avg_loss = total_loss / len(dataloader)
    return {"loss": avg_loss, "ppl": math.exp(avg_loss), "acc": total_correct / max(total_tokens, 1)}


def train_with_lb(model, trl, val_loader, cfg, device, lb_alpha=0.0, epochs=5, seed=42):
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(trl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)

    for epoch in range(1, epochs + 1):
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

    train_stats = evaluate(model, trl, device)
    val_stats = evaluate(model, val_loader, device)
    return train_stats, val_stats


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

    print(f"\nPoE params: {count(PoEModel(cfg).parameters()):,}")

    from test_load_balance import PoEModelWithLB, collect_routing_stats

    results = []
    for lb_alpha, lb_label in [(0.0, "No LB"), (0.1, "LB α=0.1")]:
        print(f"\n{'='*60}")
        print(f"  {lb_label}")
        print(f"{'='*60}")

        if lb_alpha > 0:
            model = PoEModelWithLB(cfg)
        else:
            model = PoEModel(cfg)

        train_stats, val_stats = train_with_lb(model, trl, val, cfg, device, lb_alpha, epochs=5)
        routing = collect_routing_stats(model, trl, device, num_batches=30, label=f"Routing ({lb_label})")

        results.append({
            "label": lb_label, "train": train_stats, "val": val_stats,
            "balance": routing["balance"], "entropy": routing["entropy"],
        })

    # Summary
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Condition':<15} {'Train Loss':>12} {'Val PPL':>10} {'Val Acc':>10} {'Balance':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['label']:<15} {r['train']['loss']:>12.4f} {r['val']['ppl']:>10.2f} {r['val']['acc']:>10.4f} {r['balance']:>10.3f}")

    no_lb = results[0]
    with_lb = results[1]
    ppl_diff = (no_lb['val']['ppl'] - with_lb['val']['ppl']) / no_lb['val']['ppl'] * 100
    print(f"\nPPL change with LB: {ppl_diff:+.1f}%")
    if ppl_diff > 0:
        print("→ LB loss IMPROVES performance (better balance → better PPL)")
    elif ppl_diff > -2:
        print("→ LB loss has minimal effect on PPL (balance doesn't hurt)")
    else:
        print("→ LB loss hurts performance (forcing balance is counterproductive)")


if __name__ == "__main__":
    main()
