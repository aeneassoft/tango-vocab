#!/usr/bin/env python3
"""verify_audio.py — transcribe every generated clip with Whisper and flag the ones that do not say their text.

A cheap, automatic quality gate after gen_audio.py: F5-TTS occasionally swallows a syllable or hallucinates.
Whisper (large-v3, GPU) listens to app/audio/*.m4a, the transcript is normalised like the app's answer
matching (lower-case, punctuation stripped, accents kept) and compared with the expected text using a
difflib similarity ratio. Clips below --threshold are listed; --write-regen appends their ids to
data/regen.txt so `python tools/gen_audio.py --from-regen` regenerates them with a new seed.

Usage: python tools/verify_audio.py [--only words|sentences] [--ids w12,s3] [--threshold 0.8]
                                    [--model large-v3] [--write-regen] [--limit N]
Runs inside .venv automatically (same re-exec as gen_audio.py).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gen_audio import ensure_venv, find_ffmpeg, parse_item_ids, utf8_console  # noqa: E402
from validate_data import norm  # noqa: E402

AUDIO_DIR = ROOT / "app" / "audio"
REGEN = ROOT / "data" / "regen.txt"


def load_expected() -> dict[str, str]:
    """clip name (w_12 / s_3) -> expected spoken text."""
    exp: dict[str, str] = {}
    with (ROOT / "data" / "words.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            exp[f"w_{int(r['id'])}"] = r["spanish"]
    with (ROOT / "data" / "sentences.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            exp[f"s_{int(r['id'])}"] = r["spanish"]
    return exp


UNITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
         "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte"]
TENS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"}


HUNDREDS = {100: "cien", 200: "doscientos", 300: "trescientos", 400: "cuatrocientos", 500: "quinientos", 600: "seiscientos",
            700: "setecientos", 800: "ochocientos", 900: "novecientos"}


def number_words(n: int) -> str:
    """0-999999 as spoken in Spanish (Whisper writes digits, the CSVs spell numbers out)."""
    if n < 0:
        return str(n)
    if n <= 20:
        return UNITS[n]
    if n < 30:
        return "veinti" + UNITS[n - 20]
    if n < 100:
        t, u = divmod(n, 10)
        return TENS[t * 10] + (f" y {UNITS[u]}" if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        head = "ciento" if (h == 1 and r) else HUNDREDS[h * 100]
        return head + (f" {number_words(r)}" if r else "")
    if n < 1000000:
        k, r = divmod(n, 1000)
        head = "mil" if k == 1 else f"{number_words(k)} mil"
        return head + (f" {number_words(r)}" if r else "")
    if n == 1000000:
        return "un millón"
    return str(n)


def spell_numbers(s: str) -> str:
    import re as _re
    # '300.000' / '1.500' (thousands separators) -> plain digits, then digits -> words
    sep = _re.compile('([0-9])[.,]([0-9]{3})(?![0-9])')
    while sep.search(s):
        s = sep.sub(lambda m: m.group(1) + m.group(2), s)
    return _re.sub('[0-9]+', lambda m: number_words(int(m.group())) if int(m.group()) <= 1000000 else m.group(), s)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(spell_numbers(a)), norm(spell_numbers(b))).ratio()


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("words", "sentences"))
    ap.add_argument("--ids", metavar="LIST", help="only these clips, e.g. w12,s3 (bare number = word)")
    ap.add_argument("--threshold", type=float, default=0.8, help="similarity below this is flagged (default 0.8)")
    ap.add_argument("--model", default="large-v3", help="whisper model (default large-v3)")
    ap.add_argument("--limit", type=int, help="stop after N clips")
    ap.add_argument("--write-regen", action="store_true", help="append flagged ids to data/regen.txt")
    args = ap.parse_args(argv)
    ensure_venv(sys.argv)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg not found (needed by whisper to decode .m4a)")
        return 1
    import os
    os.environ["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")

    expected = load_expected()
    names = sorted(p.stem for p in AUDIO_DIR.glob("*.m4a"))
    if args.only:
        names = [n for n in names if n.startswith("w_" if args.only == "words" else "s_")]
    if args.ids:
        wanted = {f"{k}_{i}" for k, i in parse_item_ids(args.ids)}
        names = [n for n in names if n in wanted]
    names = [n for n in names if n in expected]
    if args.limit:
        names = names[: args.limit]
    if not names:
        print("Nothing to verify (no clips in app/audio matching the CSVs).")
        return 0

    import torch
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = whisper.load_model(args.model, device=device)
    print(f"whisper {args.model} on {device} loaded in {time.time() - t0:.1f}s — verifying {len(names)} clips")

    flagged: list[tuple[str, float, str, str]] = []
    t1 = time.time()
    for i, name in enumerate(names, 1):
        path = AUDIO_DIR / f"{name}.m4a"
        try:
            res = model.transcribe(str(path), language="es", fp16=(device == "cuda"), temperature=0.0)
            heard = res["text"].strip()
        except Exception as e:  # keep going, report at the end
            heard = f"<error: {e}>"
        score = similarity(heard, expected[name])
        mark = "  " if score >= args.threshold else "!!"
        if score < args.threshold:
            flagged.append((name, score, expected[name], heard))
        if i % 25 == 0 or score < args.threshold:
            print(f"{mark} [{i}/{len(names)}] {name:7s} {score:.2f}  »{expected[name]}«  heard: »{heard}«")
    dt = time.time() - t1
    print(f"\n{len(names)} clips in {dt:.0f}s ({dt / max(1, len(names)):.2f}s/clip) — flagged: {len(flagged)}")
    for name, score, exp, heard in sorted(flagged, key=lambda x: x[1]):
        print(f"  {name:7s} {score:.2f}  expected »{exp}«  heard »{heard}«")
    if args.write_regen and flagged:
        existing = set(REGEN.read_text(encoding="utf-8").split()) if REGEN.exists() else set()
        new = [n for n, *_ in flagged if n not in existing]
        with REGEN.open("a", encoding="utf-8") as f:
            for n in new:
                f.write(n + "\n")
        print(f"{len(new)} ids appended to data/regen.txt → python tools/gen_audio.py --from-regen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
