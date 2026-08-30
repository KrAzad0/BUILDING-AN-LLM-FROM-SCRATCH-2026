from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from src import ByteTokenizer, GPT, GPTConfig


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset split is too short for the selected block_size")
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()
    out = {}
    for name, split in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train a small GPT language model from scratch")
    p.add_argument("--data", default="data/sample.txt")
    p.add_argument("--output", default="checkpoints/minigpt.pt")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, cuda:0, ...")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"device: {device}")

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_idx = int(0.9 * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]
    minimum = args.block_size + 2
    if len(train_data) < minimum or len(val_data) < minimum:
        raise ValueError(
            f"Corpus is too small for block_size={args.block_size}. "
            f"Need at least ~{10 * minimum} byte tokens for a 90/10 split."
        )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"parameters: {model.num_parameters():,}")
    print(f"training tokens: {len(train_data):,} | validation tokens: {len(val_data):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    best_val = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, args, device)
            print(
                f"step {step:6d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config.to_dict(),
                        "step": step,
                        "val_loss": best_val,
                    },
                    output,
                )
                print(f"saved checkpoint → {output}")

        if step == args.steps:
            break

        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("training complete")


if __name__ == "__main__":
    main()
