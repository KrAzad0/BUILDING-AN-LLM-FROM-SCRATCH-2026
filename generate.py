from __future__ import annotations

import argparse

import torch

from src import ByteTokenizer, GPT, GPTConfig
from train import choose_device


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a trained MiniGPT checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/minigpt.pt")
    p.add_argument("--prompt", default="The future of intelligence")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [32]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
