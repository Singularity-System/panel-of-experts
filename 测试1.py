
import torch
from alpha_eai import PoEModel, PoEConfig
from transformers import GPT2Tokenizer

cfg = PoEConfig(num_experts=2, expert_num_layers=2, post_processing_num_layers=2, d_model=128, n_head=2, d_ff=256, top_k=1, max_seq_len=128)
model = PoEModel(cfg)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
text = 'Once upon a time'
ids = tokenizer.encode(text, return_tensors='pt')

with torch.no_grad():
    out = model(ids)
    logits = out['logits']
    next_token = logits[0, -1].argmax().item()
    print('Generated next token:', tokenizer.decode(next_token))
    print('Success! Alpha EAI can generate text.')
