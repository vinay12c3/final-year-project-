"""Makes the short "how to sign it" video clips shown in the Sign guide.

For every sentence (ISL-CSLTR) and word (INCLUDE Greetings) it takes one
example video, shrinks it to a small silent MP4 plus a poster image, and writes
frontend/public/clips/manifest.json. Run once after downloading the datasets:

    python -m scripts.make_sign_clips
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

import av
import cv2

from signvision import config as C
from signvision import data as D
from signvision.sentences import label_for, text_for

OUT = C.FRONTEND_DIST.parent / "public" / "clips"
CSLTR_ZIP = C.DATA_DIR / "raw" / "isl_csltr" / "ISL_CSLRT_Corpus.zip"
INCLUDE_DIR = C.DATA_DIR / "raw" / "include" / "extracted" / "Greetings"
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")


def transcode(src: Path, dst: Path, width: int, fps: int, max_seconds: float) -> bool:
    """Small H.264 MP4 (plays in every browser) + JPEG poster from the middle frame."""
    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while len(frames) < max_seconds * src_fps:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 5:
        return False
    step = max(1, round(src_fps / fps))
    frames = frames[::step]
    h0, w0 = frames[0].shape[:2]
    h = int(round(h0 * width / w0 / 2) * 2)
    out = av.open(str(dst), "w", options={"movflags": "+faststart"})
    st = out.add_stream("libx264", rate=Fraction(fps))
    st.width, st.height, st.pix_fmt = width, h, "yuv420p"
    st.options = {"crf": "27", "preset": "veryslow", "profile": "main"}
    for f in frames:
        rgb = cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for pk in st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            out.mux(pk)
    for pk in st.encode(None):
        out.mux(pk)
    out.close()
    mid = cv2.resize(frames[len(frames) // 2], (width, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst.with_suffix(".jpg")), mid, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return True


def sentence_sources(z: zipfile.ZipFile) -> dict[str, str]:
    """Sentence folder -> first video inside it."""
    found: dict[str, str] = {}
    for n in sorted(z.namelist()):
        parts = n.split("/")
        if "Videos_Sentence_Level" in parts and n.lower().endswith(VIDEO_EXT) and len(parts) >= 3:
            found.setdefault(parts[-2], n)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--sentences-dir", help="a folder with one sub-folder of videos per sentence "
                                             "(instead of reading the ISL-CSLTR zip)")
    args = ap.parse_args()
    (OUT / "sentences").mkdir(parents=True, exist_ok=True)
    (OUT / "words").mkdir(parents=True, exist_ok=True)
    manifest = {"sentences": [], "words": []}

    def add_sentence(folder: str, get_video):
        label = label_for(folder)
        dst = OUT / "sentences" / f"{label}.mp4"
        if not dst.exists() and not transcode(get_video(), dst, args.width, args.fps, args.max_seconds):
            print("  skipped (unreadable):", folder)
            return
        manifest["sentences"].append({"label": label, "text": text_for(folder),
                                      "video": f"/clips/sentences/{label}.mp4",
                                      "poster": f"/clips/sentences/{label}.jpg"})
        print(f"  sentence: {text_for(folder)}", flush=True)

    if args.sentences_dir:
        for folder in sorted(p for p in Path(args.sentences_dir).iterdir() if p.is_dir()):
            vids = sorted(v for v in folder.iterdir() if v.suffix.lower() in VIDEO_EXT)
            if vids:
                add_sentence(folder.name, lambda v=vids[0]: v)
    elif CSLTR_ZIP.exists() and zipfile.is_zipfile(CSLTR_ZIP):
        with zipfile.ZipFile(CSLTR_ZIP) as z, tempfile.TemporaryDirectory() as tmp:
            for folder, member in sentence_sources(z).items():
                label = label_for(folder)
                dst = OUT / "sentences" / f"{label}.mp4"
                if not dst.exists():
                    src = Path(tmp) / Path(member).name
                    with z.open(member) as fi, open(src, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    ok = transcode(src, dst, args.width, args.fps, args.max_seconds)
                    src.unlink()
                    if not ok:
                        print("  skipped (unreadable):", folder)
                        continue
                manifest["sentences"].append({"label": label, "text": text_for(folder),
                                              "video": f"/clips/sentences/{label}.mp4",
                                              "poster": f"/clips/sentences/{label}.jpg"})
                print(f"  sentence: {text_for(folder)}")
    else:
        print("ISL-CSLTR not found (or still downloading) at", CSLTR_ZIP)

    if INCLUDE_DIR.exists():
        for folder in sorted(p for p in INCLUDE_DIR.iterdir() if p.is_dir()):
            label = D.clean_label(folder.name, "dynamic")
            vids = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)
            dst = OUT / "words" / f"{label}.mp4"
            if vids and (dst.exists() or transcode(vids[0], dst, args.width, args.fps, args.max_seconds)):
                manifest["words"].append({"label": label, "text": C.display_label(label),
                                          "video": f"/clips/words/{label}.mp4",
                                          "poster": f"/clips/words/{label}.jpg"})
                print(f"  word: {label}")

    manifest["sentences"].sort(key=lambda s: s["text"].lower())
    manifest["credits"] = {
        "sentences": "ISL-CSLTR, Elakkiya R. & Natarajan B. (2021), Mendeley Data, doi:10.17632/kcmpdxky7p.1, CC BY 4.0",
        "words": "INCLUDE, Sridhar et al. (2020), Zenodo, doi:10.5281/zenodo.4010759, CC BY 4.0",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"{len(manifest['sentences'])} sentences, {len(manifest['words'])} words, {size:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
