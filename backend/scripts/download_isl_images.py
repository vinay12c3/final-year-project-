"""Download a public ISL alphabet/number image dataset and unpack it into folders.

Source: Hugging Face mirror "Hemg/Indian_sign_language_dataset" of the widely
used Kaggle ISL dataset (35 classes: 1-9 and A-Z, ~42k images, ~290 MB).
Cite the original dataset in the paper; check its licence before redistributing.

Run from the backend folder:
    python -m scripts.download_isl_images
    python -m scripts.prepare_images --src data/raw/isl_images --name isl_public
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

from signvision.config import DATA_DIR
from signvision.trainer import log

URL = ("https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset/resolve/main/"
       "data/train-00000-of-00001-a1731e778755d263.parquet")
NAMES = [str(i) for i in range(1, 10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total and done % (25 << 20) < (1 << 20):
                log(f"  {done / total:5.1%} of {total / 1e6:.0f} MB")
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA_DIR / "raw" / "isl_images"))
    args = ap.parse_args()

    import pandas as pd

    parquet = DATA_DIR / "raw" / "isl_images.parquet"
    log("downloading ISL image dataset (~290 MB)...")
    download(URL, parquet)
    df = pd.read_parquet(parquet)
    out = Path(args.out)
    log(f"unpacking {len(df)} images to {out}")
    for i, row in enumerate(df.itertuples(index=False)):
        img, label = row.image, NAMES[int(row.label)]
        data = img["bytes"] if isinstance(img, dict) else img
        ext = ".png" if data[:4] == b"\x89PNG" else ".jpg"
        folder = out / label
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{i:06d}{ext}").write_bytes(io.BytesIO(data).getvalue())
    log("done. Next: python -m scripts.prepare_images --src data/raw/isl_images --name isl_public")


if __name__ == "__main__":
    main()
