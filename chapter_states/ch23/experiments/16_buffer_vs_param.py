"""Experiment 16 — Buffers vs parameters: what each one does.

Builds a tiny module with one of each and inspects how PyTorch tracks them.
"""

import torch
import torch.nn as nn


class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))     # learnable
        self.register_buffer("offset", torch.ones(2))  # not learnable, but module-owned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight + self.offset + x


def main() -> None:
    m = Demo()

    print("named_parameters:")
    for n, p in m.named_parameters():
        print(f"  {n}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
    print()

    print("named_buffers:")
    for n, b in m.named_buffers():
        print(f"  {n}: shape={tuple(b.shape)}, requires_grad={b.requires_grad}")
    print()

    # The optimiser sees only parameters
    n_optim_params = len(list(m.parameters()))
    print(f"len(list(m.parameters())) = {n_optim_params}  # only the learnable bit")
    n_state = len(m.state_dict())
    print(f"len(m.state_dict())       = {n_state}  # parameters AND buffers — both saved")


if __name__ == "__main__":
    main()
