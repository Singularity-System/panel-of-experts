"""PPoT vs Standard Transformer. 2000 synthetic samples, fair comparison."""
import os
import torch
import math
import random
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
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


def generate_synthetic_data(num_samples=2000, seed=42):
    rng = random.Random(seed)
    texts = []
    for i in range(num_samples):
        pat = i % 3
        length = 80 + rng.randint(0, 40)
        if pat == 0:
            tokens = [rng.randint(100, 200) for _ in range(length)]
        elif pat == 1:
            tokens = [rng.randint(200, 300) + (i % 3) * 50 for i in range(length)]
        else:
            tokens = [rng.randint(300, 500) if j % 5 != 0 else 100 + (j % 4) * 100 for j in range(length)]
        texts.append(" ".join(str(t) for t in tokens))
    return texts


class StandardTransformer(torch.nn.Module):
    """Standard Transformer from scratch — accurate param counting."""
    def __init__(self, vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=8, max_seq_len=128):
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


def train_model(model, trl, epochs, device, seed=0, lb_alpha=0.0, div_alpha=0.0):
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
            total_correct += ((sl.argmax(-1) == slb) & (sm == 1)).sum().item()
            total_tokens += sm.sum().item()
    avg = total_loss / len(dataloader)
    return {"loss": avg, "ppl": math.exp(avg), "acc": total_correct / max(total_tokens, 1)}


def routing_stats(model, dataloader, device, num_batches=30):
    hook = {"i": None}
    def hook_fn(m, i, o):
        hook["i"] = o[1].detach()
    h = model.router.register_forward_hook(hook_fn)
    counts = torch.zeros(model.num_experts, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            for k in range(model.top_k):
                flat = hook["i"][:, :, k].reshape(-1)
                mask = batch["attention_mask"].reshape(-1)
                valid = flat[mask.bool()]
                for e in range(model.num_experts):
                    counts[e] += (valid == e).sum().item()
    h.remove()
    total = counts.sum().item()
    balance = counts.min().item() / max(counts.max().item(), 1)
    return {"counts": counts.tolist(), "balance": balance, "total": total}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_samples = int(os.environ.get("NUM_SAMPLES", "2000"))
    texts = generate_synthetic_data(num_samples)
    print(f"Data: {len(texts)} samples")

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
    print(f"Split: {len(tr)} train, {len(va)} val")

    # PPoT config
    ppo_cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=2,
                        d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=128,
                        batch_size=16, num_epochs=5, learning_rate=3e-4,
                        lb_loss_weight=0.1, div_loss_weight=0.5)
    ppo_model = PoEModel(ppo_cfg)
    total_params = count(ppo_model.parameters())
    embed_params = sum(p.numel() for n, p in ppo_model.named_parameters() if 'wte' in n or 'wpe' in n)
    trans_params = total_params - embed_params
    print(f"\nPPoT: total={total_params:,}, embedding={embed_params:,}, transformer={trans_params:,}")

    # Transformer configs for fair comparison
    print(f"\n--- Transformer param counts ---")
    tf_configs = [
        (5, 128, "5L, d=128"),    # same effective depth as PPoT
        (8, 128, "8L, d=128"),    # moderate
        (16, 128, "16L, d=128"),  # match total params
    ]
    for nl, dm, label in tf_configs:
        m = StandardTransformer(vocab_size=50257, d_model=dm, n_head=4, d_ff=256, num_layers=nl, max_seq_len=128)
        tp = count(m.parameters())
        te = sum(p.numel() for n, p in m.named_parameters() if 'wte' in n or 'wpe' in n)
        print(f"  Transformer({label}): total={tp:,}, embedding={te:,}, transformer={tp-te:,}")

    # Train PPoT
    print(f"\n{'='*60}")
    print(f"  PPoT (LB+Div)")
    print(f"{'='*60}")
    ppo_model = PoEModel(ppo_cfg)
    train_model(ppo_model, trl, epochs=5, device=device, seed=42, lb_alpha=0.1, div_alpha=0.5)
    ppo_res = evaluate(ppo_model, val, device)
    ppo_stats = routing_stats(ppo_model, trl, device)
    print(f"PPL={ppo_res['ppl']:.2f}, Acc={ppo_res['acc']:.4f}, Balance={ppo_stats['balance']:.3f}")
    print(f"Expert counts: {ppo_stats['counts']}")

    # Train Transformers
    tf_results = []
    for nl, dm, label in tf_configs:
        print(f"\n{'='*60}")
        print(f"  Transformer({label})")
        print(f"{'='*60}")
        m = StandardTransformer(vocab_size=50257, d_model=dm, n_head=4, d_ff=256, num_layers=nl, max_seq_len=128)
        tp = count(m.parameters())
        print(f"Params: {tp:,}")
        train_model(m, trl, epochs=5, device=device, seed=42)
        res = evaluate(m, val, device)
        tf_results.append((label, res, tp))
        print(f"PPL={res['ppl']:.2f}, Acc={res['acc']:.4f}")

    # Summary
    print(f"\n{'='*75}")
    print(f"  RESULT: PPoT vs Transformer")
    print(f"{'='*75}")
    print(f"{'Model':<25} {'PPL':>8} {'Acc':>8} {'Params':>10}")
    print("-"*70)
    print(f"{'PPoT (LB+Div)':<25} {ppo_res['ppl']:>8.2f} {ppo_res['acc']:>8.4f} {total_params:>10,}")
    for label, res, tp in tf_results:
        print(f"{'Transformer '+label:<25} {res['ppl']:>8.2f} {res['acc']:>8.4f} {tp:>10,}")

    print(f"\n{'='*75}")
    print(f"  VS PPoT")
    print(f"{'='*75}")
    for label, res, tp in tf_results:
        ppl_delta = (res['ppl'] - ppo_res['ppl']) / ppo_res['ppl'] * 100
        print(f"Transformer {label}: PPL {ppl_delta:+.1f}% vs PPoT (balance={ppo_stats['balance']:.3f})")


if __name__ == "__main__":
    main()
