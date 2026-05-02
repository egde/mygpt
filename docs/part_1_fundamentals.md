---
title: Part I — LLM Fundamentals
parent: myGPT
nav_order: 2
has_children: true
---

# Part I — LLM Fundamentals

This is the original 18-chapter walkthrough. By the end you have built a working `mygpt` Python package + CLI that trains a tiny GPT-2-level character-level language model from scratch in PyTorch and generates text. Every line of code is introduced and explained; every "Expected output" block is captured from a real run.

---

## Audience and prerequisites

This part is written for university students. It assumes:

- working Python knowledge (functions, classes, modules, virtual environments),
- basic linear algebra (vectors, matrices, matrix multiplication, dot products),
- basic calculus (derivatives, partial derivatives, the chain rule),
- basic probability (mean, variance, probability distributions),
- **no prior machine learning knowledge**.

You do not need to know what gradient descent is, what an embedding is, or what a transformer is. We build all of that from scratch.

**Hardware:** CPU is enough. A typical training run finishes in under one minute.

---

## How to read this part

Each chapter follows the same shape:

1. **Concept** — the theory and intuition, with diagrams and worked examples.
2. **Math** — the equations that make the concept precise, written in $\LaTeX$.
3. **Code** — the exact files to create, with their full content.
4. **Run** — the `uv` command to execute it and what to expect in the output.
5. **Experiments** — small variations you can run to see the concept move.
6. **Exercises** — problems you solve to deepen understanding.

Always run code from the project root. The package layout we choose makes that the only place where imports work cleanly.

---

## Chapters

**Foundations**

1. [What is a language model?](01_what_is_a_language_model.html)
2. [Project setup with `uv`](02_project_setup.html)
3. [PyTorch in 20 minutes: tensors, autograd, modules](03_pytorch_primer.html)
4. [How machines learn: loss, gradients, gradient descent](04_how_machines_learn.html)
5. [From text to numbers: tokens and embeddings](05_tokens_and_embeddings.html)

**The attention mechanism**

6. [Single-head self-attention from scratch](06_self_attention.html)
7. [A reusable attention module](07_reusable_attention.html)
8. [Multi-head attention](08_multi_head_attention.html)

**The transformer**

9. [The feed-forward network and residual connections](09_mlp_and_residuals.html)
10. [Layer normalization](10_layer_norm.html)
11. [Putting it together: the transformer block](11_transformer_block.html)

**The GPT model**

12. [Position embeddings and the language modeling head](12_positions_and_head.html)
13. [The forward pass with loss](13_forward_pass_with_loss.html)
14. [Training loop: gradient descent in practice](14_training_loop.html)
15. [Generation: sampling text from a trained model](15_generation.html)

**A real workflow**

16. [A reusable character tokenizer](16_character_tokenizer.html)
17. [Training on a real text file](17_training_on_real_text.html)
18. [Checkpoints, inference, and a CLI](18_cli_and_checkpoints.html)

---

Ready? Start with [Chapter 1: What is a language model?](01_what_is_a_language_model.html). When you finish Chapter 18, continue with [Part II — Advanced Topics](part_2_advanced.html).
