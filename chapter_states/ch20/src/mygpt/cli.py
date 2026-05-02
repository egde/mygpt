import torch

from mygpt.checkpoint import load_checkpoint, save_checkpoint
from mygpt.generate import generate
from mygpt.model import GPT
from mygpt.tokenizer import CharTokenizer
from mygpt.utils import get_batch, pick_device, set_seed


def _train_command(args) -> None:
    device = pick_device(args.device)

    with open(args.text_file) as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode(text).to(device)

    set_seed(0)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device:       {device}")
    print(f"precision:    {args.precision}")
    print(f"corpus chars: {len(text):,}")
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")

    set_seed(42)
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.seq_len)
        optimizer.zero_grad()
        if args.precision == "bf16":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.print_every == 0 or step == args.steps:
            print(f"step {step:>5}: loss = {loss.item():.4f}")

    save_checkpoint(model, tokenizer, args.output)
    print(f"\nsaved checkpoint to {args.output}")


def _generate_command(args) -> None:
    device = pick_device(args.device)
    print(f"device: {device}\n")
    model, tokenizer = load_checkpoint(args.checkpoint)
    model.to(device)
    set_seed(args.seed)
    prompt = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    if args.precision == "bf16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = generate(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
    else:
        out = generate(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(out[0]))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mygpt",
        description="Tiny GPT trainer and text generator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a GPT on a plain-text file.")
    p_train.add_argument("text_file", help="Path to a UTF-8 text file.")
    p_train.add_argument("--output", default="model.ckpt", help="Checkpoint output path.")
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--seq-len", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--embed-dim", type=int, default=64)
    p_train.add_argument("--num-heads", type=int, default=4)
    p_train.add_argument("--num-layers", type=int, default=4)
    p_train.add_argument("--max-seq-len", type=int, default=64)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--print-every", type=int, default=500)
    p_train.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda → mps → cpu in that order.",
    )
    p_train.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Forward-pass precision. fp32 (default) is bit-deterministic; bf16 uses torch.autocast.",
    )
    p_train.set_defaults(func=_train_command)

    p_gen = sub.add_parser("generate", help="Generate text from a checkpoint.")
    p_gen.add_argument("--checkpoint", required=True)
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max-new-tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=10)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda → mps → cpu in that order.",
    )
    p_gen.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Forward-pass precision. fp32 (default) is bit-deterministic; bf16 uses torch.autocast.",
    )
    p_gen.set_defaults(func=_generate_command)

    args = parser.parse_args()
    args.func(args)
