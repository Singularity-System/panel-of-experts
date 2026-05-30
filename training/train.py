import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import math


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        total_loss += loss.item()

        # Token accuracy (skip padding, skip last position)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        preds = shift_logits.argmax(dim=-1)
        correct = ((preds == shift_labels) & (shift_mask == 1)).sum().item()
        tokens = shift_mask.sum().item()

        total_correct += correct
        total_tokens += tokens

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / max(total_tokens, 1)
    return {"loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy}


def train(model, dataloader, config, eval_dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_steps = len(dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Router load balancing loss (Switch Transformer style)
    lb_weight = getattr(config, "lb_loss_weight", 0.0)
    # Expert diversity loss (von Neumann entropy)
    div_weight = getattr(config, "div_loss_weight", 0.0)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            # Apply LB loss if model supports it
            if lb_weight > 0 and hasattr(model, "auxiliary_load_balance_loss"):
                lb_loss = model.auxiliary_load_balance_loss()
                loss = loss + lb_weight * lb_loss

            # Apply diversity loss if enabled
            if div_weight > 0 and hasattr(model, "expert_diversity_loss"):
                div_loss = model.expert_diversity_loss()
                loss = loss + div_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += outputs["loss"].item()
            pbar.set_postfix({"loss": f"{outputs['loss'].item():.4f}"})

        avg_loss = total_loss / len(dataloader)

        if eval_dataloader is not None:
            metrics = evaluate(model, eval_dataloader, device)
            print(f"Epoch {epoch} | loss: {avg_loss:.4f} | ppl: {metrics['perplexity']:.2f} | acc: {metrics['accuracy']:.4f}")
        else:
            print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    return model
