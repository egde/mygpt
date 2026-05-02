"""Experiment 01 — Hello from mygpt.

Confirms the package is installed and importable from a script outside
`src/mygpt/`. Prints the vocabulary along with each token's integer id.
"""

from mygpt import VOCAB


def main() -> None:
    print(f"Vocabulary size V = {len(VOCAB)}")
    for token_id, token in enumerate(VOCAB):
        print(f"  id {token_id}: {token!r}")


if __name__ == "__main__":
    main()
