"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

This file holds the package-level constants used in every chapter.
For now there is only one: the four tokens that form our running example.
"""

VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
