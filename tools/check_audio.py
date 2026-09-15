#!/usr/bin/env python3
"""check_audio.py — listen to random clips from app/audio and flag the bad ones.

Plays N random clips with ffplay, shows the Spanish text and the German meaning
and asks  ok / neu / r(eplay) / q(uit).  "neu" appends the id (w_412 / s_12) to
data/regen.txt, which  python tools/gen_audio.py --from-regen  regenerates.

  python tools/check_audio.py [--n 10] [--only words|sentences] [--ids w12,s3] [--seed N]

Standard library only + ffplay (looked up next to ffmpeg); no venv required.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AUDIO_DIR = ROOT / "app" / "audio"
WORDS_CSV = DATA_DIR / "words.csv"
SENTS_CSV = DATA_DIR / "sentences.csv"
REGEN_TXT = DATA_DIR / "regen.txt"
CLIP_RE = re.compile(r"^([ws])_(\d+)\.m4a$")

# Shared helpers live in gen_audio.py (stdlib-only at import time); fall back to
# minimal versions so this tool keeps working on its own.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from gen_audio import find_ffplay, parse_item_ids, utf8_console
except Exception:  # pragma: no cover - only when gen_audio.py is missing/broken
    def utf8_console() -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    def find_ffplay() -> str | None:
        return shutil.which("ffplay")

    def parse_item_ids(spec: str, default_kind: str = "w") -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for tok in re.split(r"[,;\s]+", spec.strip()):
            m = re.match(r"^(?:([ws])_?)?(\d+)$", tok, re.IGNORECASE) if tok else None
            if not m:
                if tok:
                    raise SystemExit(f"Bad item id {tok!r} — use 12, w12, s3, w_12 or s_3")
                continue
            key = ((m.group(1) or default_kind).lower(), int(m.group(2)))
            if key not in out:
                out.append(key)
        return out


def load_texts() -> dict[tuple[str, int], tuple[str, str]]:
    """(kind, id) -> (spanish, german) from the two CSVs (missing files are tolerated)."""
    texts: dict[tuple[str, int], tuple[str, str]] = {}
    for kind, path in (("w", WORDS_CSV), ("s", SENTS_CSV)):
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    iid = int((row.get("id") or "").strip())
                except ValueError:
                    continue
                texts[(kind, iid)] = ((row.get("spanish") or "").strip(), (row.get("german") or "").strip())
    return texts


def list_clips(only: str | None) -> list[tuple[str, int, Path]]:
    clips: list[tuple[str, int, Path]] = []
    if not AUDIO_DIR.is_dir():
        return clips
    for p in sorted(AUDIO_DIR.glob("*.m4a")):
        m = CLIP_RE.match(p.name)
        if not m:
            continue
        kind = m.group(1)
        if (only == "words" and kind != "w") or (only == "sentences" and kind != "s"):
            continue
        clips.append((kind, int(m.group(2)), p))
    return clips


def read_regen() -> list[str]:
    if not REGEN_TXT.exists():
        return []
    return [ln.split("#", 1)[0].strip() for ln in REGEN_TXT.read_text(encoding="utf-8-sig").splitlines()
            if ln.split("#", 1)[0].strip()]


def add_regen(name: str) -> bool:
    """Append `name` to data/regen.txt unless it is already listed. -> True if added."""
    if name in read_regen():
        return False
    REGEN_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REGEN_TXT.open("a", encoding="utf-8") as f:
        f.write(name + "\n")
    return True


class Player:
    def __init__(self) -> None:
        self.ffplay = find_ffplay()
        self._noted = False

    def play(self, path: Path) -> None:
        if self.ffplay:
            subprocess.run([self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
            return
        if not self._noted:
            self._noted = True
            if os.name == "nt":
                print("note: ffplay not found — opening clips with the default Windows player instead "
                      "(winget install Gyan.FFmpeg gives you ffplay)")
            else:
                print("note: ffplay not found — clips cannot be played; install ffmpeg (includes ffplay)")
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]


def ask() -> str:
    try:
        return input("ok / neu / r(eplay) / q(uit) [ok]: ").strip().lower()
    except EOFError:
        print()
        return "q"


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        prog="check_audio.py",
        description="Play random clips from app/audio, show their text, flag bad ones for regeneration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="answers:  ok (Enter) = fine, neu = regenerate (-> data/regen.txt), r = replay, q = quit",
    )
    ap.add_argument("--n", type=int, default=10, help="number of random clips (default 10)")
    ap.add_argument("--only", choices=("words", "sentences"), help="restrict to words or sentences")
    ap.add_argument("--ids", metavar="LIST", help="check exactly these clips, e.g. w12,s3 (bare number = word)")
    ap.add_argument("--seed", type=int, help="seed for the random pick (deterministic selection)")
    args = ap.parse_args(argv)
    if args.n <= 0:
        ap.error("--n must be positive")

    clips = list_clips(args.only)
    if not clips:
        print(f"No clips in {AUDIO_DIR} — run  python tools/gen_audio.py  first.")
        return 1
    texts = load_texts()
    if not texts:
        print("note: data/words.csv / data/sentences.csv not found — texts cannot be shown")

    if args.ids:
        by_key = {(kind, iid): (kind, iid, p) for kind, iid, p in clips}
        chosen = []
        for kind, iid in parse_item_ids(args.ids):
            hit = by_key.get((kind, iid))
            if hit is None:
                print(f"warning: {kind}_{iid}.m4a not found in app/audio" + (" (or excluded by --only)" if args.only else ""))
            else:
                chosen.append(hit)
    else:
        chosen = random.Random(args.seed).sample(clips, min(args.n, len(clips)))
    if not chosen:
        print("Nothing to check.")
        return 1

    player = Player()
    ok_count, flagged = 0, []
    total = len(chosen)
    try:
        for n, (kind, iid, path) in enumerate(chosen, start=1):
            name = f"{kind}_{iid}"
            spanish, german = texts.get((kind, iid), ("?", "text unknown"))
            print(f"[{n}/{total}] {name}  »{spanish}«  ({german})")
            player.play(path)
            quit_now = False
            while True:
                ans = ask()
                if ans in ("", "ok", "o", "y", "j"):
                    ok_count += 1
                    break
                if ans in ("neu", "n", "new", "regen"):
                    added = add_regen(name)
                    flagged.append(name)
                    print(f"  -> {name} {'added to' if added else 'already in'} data/regen.txt")
                    break
                if ans in ("r", "replay"):
                    player.play(path)
                    continue
                if ans in ("q", "quit", "exit"):
                    quit_now = True
                    break
                print("  please answer ok, neu, r or q")
            if quit_now:
                break
    except KeyboardInterrupt:  # during playback (ffplay gets the same Ctrl+C) or at the prompt
        print()

    checked = ok_count + len(flagged)
    print(f"\nChecked {checked}/{total}: {ok_count} ok, {len(flagged)} flagged" + (f" ({', '.join(flagged)})" if flagged else ""))
    pending = read_regen()
    if pending:
        print(f"{len(pending)} id(s) waiting in data/regen.txt — regenerate with:\n  python tools/gen_audio.py --from-regen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
