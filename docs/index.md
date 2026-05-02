---
title: LLM Fundamentals
nav_order: 1
---

# LLM Fundamentals

A code-along tutorial that builds a GPT-2-level language model from scratch, in PyTorch, while explaining every concept along the way.

By the end, you will have a Python package called **`mygpt`** that you can train on a text file and use to generate text from the command line. Every chapter builds on the previous one. Every chapter ends with experiments and exercises that let you observe the behaviour you just read about.

The running example throughout the entire tutorial is the four tokens

```text
I  love  AI  !
```

We use these tokens over and over again, because they are short enough to inspect by hand and rich enough to demonstrate every idea — from probability distributions to attention.

---

## Audience and prerequisites

This tutorial is written for university students. It assumes:

- working Python knowledge (functions, classes, modules, virtual environments),
- basic linear algebra (vectors, matrices, matrix multiplication, dot products),
- basic calculus (derivatives, partial derivatives, the chain rule),
- basic probability (mean, variance, probability distributions),
- **no prior machine learning knowledge**.

You do not need to know what gradient descent is, what an embedding is, or what a transformer is. We build all of that from scratch.

---

## How to read this tutorial

Each chapter follows the same shape:

1. **Concept** — the theory and intuition, with diagrams and worked examples.
2. **Math** — the equations that make the concept precise, written in $\LaTeX$.
3. **Code** — the exact files to create, with their full content.
4. **Run** — the `uv` command to execute it and what to expect in the output.
5. **Experiments** — small variations you can run to see the concept move.
6. **Exercises** — problems you solve to deepen understanding.

Always run code from the project root. The package layout we choose makes that the only place where imports work cleanly.

---

## Table of contents

**Part I — Foundations**

1. [What is a language model?](01_what_is_a_language_model.md)
2. [Project setup with `uv`](02_project_setup.md)
3. [PyTorch in 20 minutes: tensors, autograd, modules](03_pytorch_primer.md)
4. [How machines learn: loss, gradients, gradient descent](04_how_machines_learn.md)
5. [From text to numbers: tokens and embeddings](05_tokens_and_embeddings.md)

**Part II — The attention mechanism**

6. [Single-head self-attention from scratch](06_self_attention.md)
7. [A reusable attention module](07_reusable_attention.md)
8. [Multi-head attention](08_multi_head_attention.md)

**Part III — The transformer**

9. [The feed-forward network and residual connections](09_mlp_and_residuals.md)
10. [Layer normalization](10_layer_norm.md)
11. [Putting it together: the transformer block](11_transformer_block.md)

**Part IV — The GPT model**

12. [Position embeddings and the language modeling head](12_positions_and_head.md)
13. [The forward pass with loss](13_forward_pass_with_loss.md)
14. [Training loop: gradient descent in practice](14_training_loop.md)
15. [Generation: sampling text from a trained model](15_generation.md)

**Part V — A real workflow**

16. [A reusable character tokenizer](16_character_tokenizer.md)
17. Training on a real text file *(coming soon)*
18. Checkpoints, inference, and a CLI *(coming soon)*

---

Ready? Start with [Chapter 1: What is a language model?](01_what_is_a_language_model.md)
