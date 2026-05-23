import argparse
import torch
from alpha_eai.config import PoEConfig, DatasetConfig
from alpha_eai.model import PoEModel
from training.dataset import make_tokenizer
from training.data_demo import make_data_loader as make_demo_loader
from training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-tiny-stories", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=== Alpha EAI - Panel of Experts MVP ===")

    poe_config = PoEConfig(
        num_experts=4,
        expert_num_layers=5,
        post_processing_num_layers=6,
        d_model=256,
        n_head=4,
        d_ff=512,
        top_k=2,
        max_seq_len=256,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=3e-4,
    )

    dataset_config = DatasetConfig(max_seq_len=256)

    print("Loading tokenizer...")
    tokenizer = make_tokenizer(dataset_config)

    if args.use_tiny_stories:
        from training.dataset import load_tiny_stories
        print("Loading TinyStories dataset...")
        dataset = load_tiny_stories(dataset_config)
        from training.dataset import make_data_loader
        train_loader = make_data_loader(dataset, tokenizer, dataset_config)
    else:
        print("Using demo dataset...")
        train_loader = make_demo_loader(tokenizer, dataset_config, poe_config.batch_size)

    print(f"Dataset loaded: {len(train_loader)} batches")

    print("Building PoE model...")
    model = PoEModel(poe_config)

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for e in model.experts for p in e.parameters())
    pp_params = sum(p.numel() for p in model.post_processing.parameters())
    print(f"Total params: {total_params:,}")
    print(f"  Experts ({poe_config.num_experts}x): {expert_params:,}")
    print(f"  Post-processing: {pp_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Starting training...")
    model = train(model, train_loader, poe_config)

    print("Training complete!")


if __name__ == "__main__":
    main()
