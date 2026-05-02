"""Experiment 17 — Dropout in train mode vs eval mode."""

import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(42)
    drop = nn.Dropout(p=0.5)
    x = torch.ones(2, 4)
    print(f"x = {x}")
    print()

    # Train mode (default after construction)
    drop.train()
    out_train = drop(x)
    print("dropout(x) in train mode (random zeros, others scaled by 1/(1-0.5) = 2):")
    print(out_train)
    print()

    # Eval mode
    drop.eval()
    out_eval = drop(x)
    print("dropout(x) in eval mode (identity — same as input):")
    print(out_eval)


if __name__ == "__main__":
    main()
