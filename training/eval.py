import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()

    return total_loss / len(dataloader)
