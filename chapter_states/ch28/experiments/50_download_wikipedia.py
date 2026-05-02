"""Download the wikitext-103-raw-v1 corpus to wikipedia.txt (Ch.28).

Fetches a pinned mirror of the Salesforce wikitext-103-raw-v1 train split,
unzips it, concatenates the train/valid/test files, strips the few wikitext
header markers that bloat the vocab without helping training, and writes the
result to ``wikipedia.txt`` in the current working directory.

The download URL is fixed (not the flaky HuggingFace dataset API); the script
is idempotent — if ``wikipedia.txt`` is already present it does nothing.

Run this once from the project root:

    uv run python experiments/50_download_wikipedia.py

Final size on disk: ~520 MB.  Time on a typical home connection: a few minutes
for the download, ~10 s for the unzip + concat.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import urllib.request
import zipfile

URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
EXPECTED_BYTES = 191_984_949  # As of 2024-03; pinned, do not adjust silently.
OUT_PATH = "wikipedia.txt"
ZIP_PATH = "wikitext-103-raw-v1.zip"
EXTRACT_DIR = "wikitext-103-raw"
TRAIN_FILE = os.path.join(EXTRACT_DIR, "wiki.train.raw")
VALID_FILE = os.path.join(EXTRACT_DIR, "wiki.valid.raw")
TEST_FILE = os.path.join(EXTRACT_DIR, "wiki.test.raw")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: str) -> None:
    print(f"downloading {url}")
    print(f"          → {dest}")
    # Some CDN mirrors (smerity.com among them) reject the default
    # ``Python-urllib/x.y`` User-Agent with 403; spoof a generic one.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100.0 * downloaded / total
                sys.stdout.write(
                    f"\r  {_human(downloaded)} / {_human(total)} ({pct:.1f}%)"
                )
                sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def _strip_wikitext_markup(text: str) -> str:
    """Strip the ``= Header =`` lines wikitext interleaves between articles.

    These lines bloat the BPE vocabulary with non-content tokens.  Everything
    else is left untouched: punctuation, casing, the ``@-@`` tokens (a wikitext
    artefact for hyphens), and the blank lines between paragraphs all survive.
    """
    return re.sub(r"^\s*=+ [^=]+ =+\s*$\n?", "", text, flags=re.MULTILINE)


def main() -> int:
    if os.path.exists(OUT_PATH):
        size = os.path.getsize(OUT_PATH)
        print(f"{OUT_PATH} already exists ({_human(size)}); nothing to do.")
        return 0

    if not os.path.exists(ZIP_PATH):
        _download(URL, ZIP_PATH)
    actual = os.path.getsize(ZIP_PATH)
    if actual != EXPECTED_BYTES:
        print(
            f"warning: {ZIP_PATH} is {actual:,} bytes; expected "
            f"{EXPECTED_BYTES:,}.  Continuing anyway.",
            file=sys.stderr,
        )

    if not os.path.exists(TRAIN_FILE):
        print(f"unzipping {ZIP_PATH} → {EXTRACT_DIR}/")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(".")

    print(f"concatenating train + valid + test → {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for src in (TRAIN_FILE, VALID_FILE, TEST_FILE):
            with open(src, encoding="utf-8") as f:
                cleaned = _strip_wikitext_markup(f.read())
            out.write(cleaned)

    out_size = os.path.getsize(OUT_PATH)
    print(f"wrote {OUT_PATH}: {_human(out_size)}")

    # Tidy up the intermediate files; keeping them around eats ~720 MB of disk.
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    print("removed intermediate zip + extract dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
