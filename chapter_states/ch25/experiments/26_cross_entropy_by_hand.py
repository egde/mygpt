"""Experiment 26 — Cross-entropy by hand on (logits=(1.0, 2.0, 0.5, -1.0), target=1)."""

import math

import torch
import torch.nn.functional as F


def main() -> None:
    logits = torch.tensor([1.0, 2.0, 0.5, -1.0])
    target = torch.tensor(1)

    # By hand
    exp_logits = torch.exp(logits)
    sum_exp = exp_logits.sum().item()
    softmax_target = exp_logits[target].item() / sum_exp
    loss_by_hand = -math.log(softmax_target)

    print(f"logits:                   {logits}")
    print(f"target:                   {target.item()}")
    print(f"exp(logits):              {exp_logits}")
    print(f"sum exp:                  {sum_exp:.6f}")
    print(f"softmax[target]:          {softmax_target:.6f}")
    print(f"-log(softmax[target]):    {loss_by_hand:.6f}")
    print()

    # By F.cross_entropy
    loss_torch = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    print(f"F.cross_entropy:          {loss_torch.item():.6f}")
    print(f"matches by-hand:          {abs(loss_by_hand - loss_torch.item()) < 1e-6}")


if __name__ == "__main__":
    main()
