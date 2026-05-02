"""Experiment 31 — Temperature reshapes the softmax distribution.

For logits=(2.0, 1.0, 0.0), show softmax at several temperatures.
"""

import torch
import torch.nn.functional as F


def main() -> None:
    logits = torch.tensor([2.0, 1.0, 0.0])
    print(f"logits: {logits}\n")
    for tau in (1.0, 0.5, 2.0, 100.0):
        probs = F.softmax(logits / tau, dim=-1)
        print(f"  tau = {tau:>5.1f}: softmax = {[f'{p:.3f}' for p in probs.tolist()]}")


if __name__ == "__main__":
    main()
