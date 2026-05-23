from dataclasses import dataclass


@dataclass
class PoEConfig:
    num_experts: int = 4
    expert_num_layers: int = 5
    post_processing_num_layers: int = 6
    d_model: int = 256
    n_head: int = 4
    d_ff: int = 512
    top_k: int = 2
    vocab_size: int = 50257
    max_seq_len: int = 256
    expert_variant: str = "gpt2"

    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05


@dataclass
class DatasetConfig:
    name: str = "the_cool_kid/tiny_stories"
    split: str = "train"
    max_seq_len: int = 256
    num_workers: int = 4
