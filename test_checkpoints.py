"""
Load saved checkpoints and evaluate PPoT vs Transformer on wikitext-103.

Usage:
    python3 test_checkpoints.py
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


class StandardTransformer(torch.nn.Module):
    """Standard Transformer from scratch."""
    def __init__(self, vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=5, max_seq_len=256):
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


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks), "labels": torch.stack(padded).clone()}


def evaluate(model, dataloader, device):
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


def count_layers(state_dict):
    """Detect number of transformer layers from state_dict keys."""
    max_layer = 0
    for k in state_dict.keys():
        if "layers." in k:
            layer_idx = int(k.split("layers.")[1].split(".")[0])
            max_layer = max(max_layer, layer_idx)
    return max_layer + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-103")
    parser.add_argument("--samples", type=int, default=50000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    cache_dir = "."
    search_order = [f"{args.dataset}-raw", args.dataset]
    data_path = None
    for name in search_order:
        path = os.path.join(cache_dir, name, "wiki.train.raw")
        if os.path.exists(path):
            data_path = path
            break

    if not data_path:
        print(f"[Error] Wikitext data not found!")
        return

    with open(data_path, "r") as f:
        lines = [s.strip() for s in f if len(s.strip()) > 20][:args.samples]
    print(f"[Data] Loaded {len(lines)} lines from {data_path}")

    tokenizer = make_tokenizer(type("C", (), {"vocab_size": 50257})())
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, tok, ms):
            self.texts = texts; self.tok = tok; self.ms = ms
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.tok(self.texts[i], return_tensors="pt", max_length=self.ms, truncation=True)["input_ids"].squeeze(0)

    ds = DS(lines, tokenizer, 256)
    tr, va = torch.utils.data.random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(tr)} train, {len(va)} val")

    # ============================================================
    # Load and evaluate Transformer
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Transformer")
    print(f"{'='*70}")

    tf_ckpt = "model_checkpoints/transformer.pt"
    if not os.path.exists(tf_ckpt):
        print(f"[Error] Not found: {tf_ckpt}")
        return

    tf_state = torch.load(tf_ckpt, map_location=device)
    tf_nl = count_layers(tf_state)
    print(f"[Info] Detected {tf_nl} layers from checkpoint")

    tf_model = StandardTransformer(vocab_size=50257, d_model=128, n_head=4, d_ff=256, num_layers=tf_nl, max_seq_len=256)
    tf_model.load_state_dict(tf_state)
    print(f"[OK] Loaded {tf_ckpt}")

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

    ppo_ckpt = "model_checkpoints/ppot.pt"
    if not os.path.exists(ppo_ckpt):
        print(f"[Error] Not found: {ppo_ckpt}")
        return

    cfg = PoEConfig(num_experts=4, expert_num_layers=3, post_processing_num_layers=3,
                    d_model=128, n_head=4, d_ff=256, top_k=2, max_seq_len=256,
                    batch_size=16, num_epochs=5, learning_rate=3e-4,
                    lb_loss_weight=0.1, div_loss_weight=0.5, num_gpus=2)

    ppo_model = PoEModel(cfg)
    ppo_model.load_state_dict(torch.load(ppo_ckpt, map_location=device))
    print(f"[OK] Loaded {ppo_ckpt}")

    ppo_model.to(device)
    res_ppo = evaluate(ppo_model, val, device)
    tp_ppo = sum(p.numel() for p in ppo_model.parameters())
    print(f"PPL={res_ppo['ppl']:.2f}, Acc={res_ppo['acc']:.4f}, Params={tp_ppo:,}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'PPL':>8} {'Acc':>8} {'Params':>10}")
    print("-"*70)
    print(f"{'Transformer':<20} {res_tf['ppl']:>8.2f} {res_tf['acc']:>8.4f} {tp_tf:>10,}")
    print(f"{'PPoT':<20} {res_ppo['ppl']:>8.2f} {res_ppo['acc']:>8.4f} {tp_ppo:>10,}")

    ppl_improve = (res_tf["ppl"] - res_ppo["ppl"]) / res_tf["ppl"] * 100
    acc_improve = (res_ppo["acc"] - res_tf["acc"]) / res_tf["acc"] * 100
    print(f"\nPPoT vs Transformer:")
    print(f"  PPL improvement: {ppl_improve:+.1f}%")
    print(f"  Acc improvement: {acc_improve:+.1f}%")


if __name__ == "__main__":
    main()
