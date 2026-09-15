#!/usr/bin/env python3
"""gen_audio.py — generate the Tango vocabulary clips with F5-TTS.

  data/words.csv      -> app/audio/w_<id>.m4a
  data/sentences.csv  -> app/audio/s_<id>.m4a

Per item: F5-TTS (Spanish fine-tune jpgallegoar/F5-Spanish by default, cloned from
data/reference.wav) -> trim silence -> -16 LUFS -> peak cap -1 dBFS -> ffmpeg AAC
48 kbit/s mono 24 kHz. app/audio/manifest.json lists every clip with its duration
for the service worker. Existing clips are skipped unless --force / --ids /
--from-regen is given, so adding a handful of words takes a few minutes.

Usage:
  python tools/gen_audio.py                  generate missing clips
  python tools/gen_audio.py --test           comparison clips -> data/test_clips/
  python tools/gen_audio.py --dry-run        show the plan, load nothing
  python tools/gen_audio.py --from-regen     regenerate the ids in data/regen.txt
  python tools/gen_audio.py --ids w12,s3,15  regenerate single items

The script re-executes itself inside <repo>/.venv (created by tools/setup_tts.sh);
only --help and --dry-run run outside the venv. Heavy libraries (torch, f5_tts,
whisper, librosa) are imported lazily so importing this module stays cheap.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import glob
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
DATA_DIR = ROOT / "data"
AUDIO_DIR = ROOT / "app" / "audio"
WORDS_CSV = DATA_DIR / "words.csv"
SENTS_CSV = DATA_DIR / "sentences.csv"
REF_DEFAULT = DATA_DIR / "reference.wav"
REF_SNIPPET = DATA_DIR / "reference_snippet.wav"
REF_TEXT = DATA_DIR / "reference.txt"
REF_META = DATA_DIR / "reference_meta.json"
REGEN_TXT = DATA_DIR / "regen.txt"
FAILURES_TXT = DATA_DIR / "gen_failures.txt"
TEST_DIR = DATA_DIR / "test_clips"
TMP_WAV_DIR = DATA_DIR / "tmp_wav"
MANIFEST = AUDIO_DIR / "manifest.json"

SR = 24000  # F5-TTS / vocos output rate; also the m4a rate
SPANISH_REPO = "jpgallegoar/F5-Spanish"
SPANISH_CKPT = "model_1200000.safetensors"
SPANISH_VOCAB = "vocab.txt"
WHISPER_MODEL = "large-v3"

TARGET_LUFS = -16.0
PEAK_CAP_DBFS = -1.0
PAD_START_S, PAD_END_S = 0.040, 0.080
MIN_WORD_S, MIN_SENT_S, MAX_CLIP_S, MIN_PEAK = 0.25, 0.6, 15.0, 0.01
MAX_ATTEMPTS = 3
MANIFEST_EVERY = 50
# preprocess_ref_audio_text() inside F5-TTS clips any reference longer than 12 s
# but leaves the transcript untouched, so the reference actually fed to the model
# must stay below this (the F5-TTS README also recommends "< 12 s").
F5_MAX_REF_S = 12.0
DEFAULT_REF_SECONDS = 11.0
TERMINAL_PUNCT = ".?!…"

# (file stem, kind, word mode, text) — written to data/test_clips/ by --test
TEST_ITEMS = [
    ("word_bare_calle", "w", "bare", "calle"),
    ("word_double_calle", "w", "double", "calle"),
    ("word_bare_plata", "w", "bare", "plata"),
    ("word_double_plata", "w", "double", "plata"),
    ("sentence_1", "s", "bare", "¿Vos querés bailar un tango más?"),
    ("sentence_2", "s", "bare", "La calle está llena de gente hoy."),
    ("sentence_3", "s", "bare", "Dale, che, vamos a tomar un café acá."),
]


# --------------------------------------------------------------------------- misc

def utf8_console() -> None:
    """Windows consoles default to cp1252; make »ñ« and friends printable. Line
    buffering keeps stdout in order with tqdm/stderr when the output is piped."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def die(msg: str, code: int = 1) -> NoReturn:
    sys.stdout.flush()
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


@contextlib.contextmanager
def quiet_stdout():
    """F5-TTS prints ref/gen texts unconditionally; keep the progress bar clean."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        yield sink


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- venv

def venv_python() -> Path | None:
    for cand in (VENV_DIR / "Scripts" / "python.exe", VENV_DIR / "bin" / "python"):
        if cand.exists():
            return cand
    return None


def in_venv() -> bool:
    try:
        prefix = Path(sys.prefix).resolve()
        exe = Path(sys.executable).resolve()
        venv = VENV_DIR.resolve()
    except OSError:
        return False
    return prefix == venv or exe.is_relative_to(venv)


def ensure_venv(argv: list[str]) -> None:
    """Re-exec inside <repo>/.venv when started with another interpreter."""
    if in_venv():
        return
    py = venv_python()
    if py is None:
        die(
            f"The TTS environment {rel(VENV_DIR)} does not exist.\n"
            "Create it once with:  bash tools/setup_tts.sh\n"
            "(installs torch+CUDA, f5-tts, whisper, audio helpers and checks ffmpeg)",
            code=2,
        )
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    sys.stdout.flush()
    proc = subprocess.Popen([str(py), str(Path(__file__).resolve()), *argv], env=env)
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        # The child gets the same Ctrl+C and does its own cleanup (manifest, regen.txt);
        # wait for it and pass its exit code on instead of dumping a traceback here.
        rc = proc.wait()
    sys.exit(rc)


# --------------------------------------------------------------------------- ffmpeg

def _tool_candidates(name: str) -> list[Path]:
    exe = f"{name}.exe" if os.name == "nt" else name
    cands: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        base = Path(local) / "Microsoft" / "WinGet"
        cands.append(base / "Links" / exe)
        pattern = str(base / "Packages" / "*FFmpeg*" / "**" / "bin" / exe)
        cands.extend(Path(p) for p in sorted(glob.glob(pattern, recursive=True), reverse=True))
    cands.append(Path("C:/ffmpeg/bin") / exe)
    return cands


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for cand in _tool_candidates(name):
        if cand.is_file():
            return str(cand)
    return None


def find_ffmpeg() -> str | None:
    return find_tool("ffmpeg")


def find_ffplay() -> str | None:
    found = find_tool("ffplay")
    if found:
        return found
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffplay.exe" if os.name == "nt" else "ffplay")
        if sibling.is_file():
            return str(sibling)
    return None


def require_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        die(
            "ffmpeg not found (PATH, %LOCALAPPDATA%\\Microsoft\\WinGet\\Links, WinGet Packages, C:\\ffmpeg\\bin).\n"
            "Install it with:  winget install Gyan.FFmpeg   (or: bash tools/setup_tts.sh)"
        )
    # pydub / whisper look for ffmpeg on PATH; make the discovered one visible to them.
    os.environ["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg


# --------------------------------------------------------------------------- items

@dataclass
class Item:
    kind: str  # "w" (word) | "s" (sentence)
    id: int
    text: str
    german: str = ""

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.id}"

    @property
    def out_path(self) -> Path:
        return AUDIO_DIR / f"{self.name}.m4a"


ID_RE = re.compile(r"^(?:([ws])_?)?(\d+)$", re.IGNORECASE)


def parse_item_ids(spec: str, default_kind: str = "w") -> list[tuple[str, int]]:
    """'w12,s3,15,w_7' -> [('w',12),('s',3),('w',15),('w',7)] (bare int = word)."""
    out: list[tuple[str, int]] = []
    for tok in re.split(r"[,;\s]+", spec.strip()):
        if not tok:
            continue
        m = ID_RE.match(tok)
        if not m:
            raise SystemExit(f"Bad item id {tok!r} — use 12, w12, s3, w_12 or s_3")
        key = ((m.group(1) or default_kind).lower(), int(m.group(2)))
        if key not in out:
            out.append(key)
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def load_items() -> tuple[list[Item], list[str]]:
    """Read words.csv + sentences.csv. Returns (items, notes about missing/odd rows)."""
    items: list[Item] = []
    notes: list[str] = []
    for kind, path in (("w", WORDS_CSV), ("s", SENTS_CSV)):
        if not path.exists():
            notes.append(f"{rel(path)} not found")
            continue
        for row in read_csv_rows(path):
            try:
                iid = int(row.get("id", ""))
            except ValueError:
                notes.append(f"{path.name}: skipping row with id {row.get('id')!r}")
                continue
            text = row.get("spanish", "")
            if not text:
                notes.append(f"{path.name}: id {iid} has an empty spanish field — skipped")
                continue
            items.append(Item(kind, iid, text, row.get("german", "")))
    return items, notes


def ensure_terminal_punct(text: str) -> str:
    text = " ".join(text.split())
    return text if text and text[-1] in TERMINAL_PUNCT else text + "."


def gen_text_for(kind: str, text: str, word_mode: str) -> str:
    """Sentence: as-is with terminal punctuation. Word: 'calle.' (bare) or 'calle. calle.' (double)."""
    unit = ensure_terminal_punct(text)
    if kind == "s" or word_mode == "bare":
        return unit
    return f"{unit} {unit}"


def effective_speed(kind: str, gen_text: str, speed: float, word_speed: float) -> float:
    """F5-TTS sizes the output from the byte length of the text and forces speed 0.3
    for texts under 10 bytes; a word of 10-29 bytes ('colectivo.', 'calle. calle.')
    would otherwise get a budget so tight that the onset is clipped, so --word-speed
    (default 0.7) is applied on top of --speed for those."""
    n = len(gen_text.encode("utf-8"))
    if kind == "w" and 10 <= n < 30:
        return speed * word_speed
    return speed


def seed_for(base: int, item_id: int, attempt: int) -> int:
    """Deterministic: base offset (--seed, default 0) + item id + 1000 per retry."""
    return base + item_id + attempt * 1000


def read_regen_ids() -> list[tuple[str, int]]:
    if not REGEN_TXT.exists():
        return []
    ids: list[tuple[str, int]] = []
    for line in REGEN_TXT.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            for key in parse_item_ids(line):
                if key not in ids:
                    ids.append(key)
    return ids


@dataclass
class Plan:
    work: list[Item]            # items to generate, in order
    skipped: int                # items left alone because their .m4a exists
    warnings: list[str]
    selected: bool              # an explicit --ids / --from-regen selection was given
    pending: list[str]          # --from-regen: names that stay in regen.txt unless generated now


def select_work(args, items: list[Item]) -> Plan:
    """Apply --only / --ids / --from-regen / --force to the CSV items.

    With --from-regen every listed id that exists in a CSV is 'pending': it is removed
    from regen.txt only once its clip has been generated, so ids that are excluded by
    --only, cut off by --limit, interrupted or failed stay listed. Ids that are in
    neither CSV are dropped (with a warning)."""
    warnings: list[str] = []
    by_key = {(it.kind, it.id): it for it in items}
    allowed_kind = {"words": "w", "sentences": "s"}.get(args.only)
    wanted: list[tuple[str, int]] | None = None
    if args.from_regen:
        wanted = read_regen_ids()
        if not wanted:
            warnings.append(f"{rel(REGEN_TXT)} is empty or missing — nothing to regenerate")
    elif args.ids:
        wanted = parse_item_ids(args.ids)
    pending: list[str] = []
    if wanted is not None:
        plan: list[Item] = []
        for kind, iid in wanted:
            name = f"{kind}_{iid}"
            it = by_key.get((kind, iid))
            if it is None:
                src = "words.csv" if kind == "w" else "sentences.csv"
                warnings.append(f"{name} is not in {src} — " + ("dropped from regen.txt" if args.from_regen else "skipped"))
                continue
            pending.append(name)
            if allowed_kind and it.kind != allowed_kind:
                warnings.append(f"{name} excluded by --only {args.only}" + (" — kept in regen.txt" if args.from_regen else ""))
                continue
            plan.append(it)
        force = True
    else:
        plan = [it for it in items if not allowed_kind or it.kind == allowed_kind]
        force = bool(args.force)
    work = [it for it in plan if force or not it.out_path.exists()]
    return Plan(work, len(plan) - len(work), warnings, wanted is not None, pending)


def update_regen(pending: list[str], done: set[str]) -> None:
    """Rewrite data/regen.txt with the pending ids that were not generated in this run."""
    remaining = [name for name in pending if name not in done]
    if remaining:
        REGEN_TXT.write_text("".join(f"{name}\n" for name in remaining), encoding="utf-8")
        print(f"{rel(REGEN_TXT)} still lists {len(remaining)} item(s): {', '.join(remaining)}")
    elif REGEN_TXT.exists():
        REGEN_TXT.write_text("", encoding="utf-8")
        print(f"{rel(REGEN_TXT)} cleared")


# --------------------------------------------------------------------------- audio helpers (numpy)

def load_audio_mono(path: Path, ffmpeg: str | None = None):
    """-> (float32 mono, sr) at the native rate. Falls back to ffmpeg for formats libsndfile lacks."""
    import numpy as np
    import soundfile as sf

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # e.g. .m4a / .mp4 reference
        if not ffmpeg:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", str(tmp_path)],
                           check=True)
            y, sr = sf.read(str(tmp_path), dtype="float32", always_2d=True)
        finally:
            tmp_path.unlink(missing_ok=True)
    return np.ascontiguousarray(y.mean(axis=1), dtype=np.float32), int(sr)


def rms_frames(y, sr: int, hop_s: float):
    """Non-overlapping RMS envelope. -> (rms per frame, hop in samples)."""
    import numpy as np

    hop = max(1, int(round(sr * hop_s)))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32), hop
    frames = y[: n * hop].reshape(n, hop).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1)), hop


def extract_snippet(y, sr: int, seconds: float):
    """Best speech-rich window of `seconds`: 50 ms RMS envelope, voiced = RMS above
    (peak - 25 dB), score = voiced fraction, small bonus when the window starts and
    ends inside a silent gap (>= 250 ms). 100 ms fades. -> (snippet, start_s, end_s)."""
    import numpy as np

    rms, hop = rms_frames(y, sr, 0.050)
    n = len(rms)
    win = max(1, int(round(seconds / 0.050)))
    if n <= win:
        return y.copy(), 0.0, len(y) / sr
    voiced = rms > rms.max() * db_to_lin(-25.0)
    gap = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if voiced[i]:
            i += 1
            continue
        j = i
        while j < n and not voiced[j]:
            j += 1
        if j - i >= 5:  # 5 * 50 ms = 250 ms
            gap[i:j] = True
        i = j
    csum = np.concatenate([[0], np.cumsum(voiced)])
    best_start, best_score = 0, -1.0
    for s in range(0, n - win + 1):
        e = s + win
        score = (csum[e] - csum[s]) / win + 0.02 * (int(gap[s]) + int(gap[e - 1]))
        if score > best_score:
            best_score, best_start = score, s
    start = best_start * hop
    end = min(len(y), (best_start + win) * hop)
    snip = y[start:end].astype(np.float32).copy()
    fade = min(int(sr * 0.100), len(snip) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        snip[:fade] *= ramp
        snip[-fade:] *= ramp[::-1]
    return snip, start / sr, end / sr


def keep_second_utterance(y, sr: int):
    """Double mode: the text was '<word>. <word>.'; keep only the second utterance.

    1. Prefer the longest internal gap (>= 120 ms below peak - 30 dB) whose centre lies in the
       middle 30-70 % of the clip.
    2. Otherwise cut at the softest point of a smoothed envelope in that window (if it is at
       least 18 dB below the peak) - F5 often leaves only a short dip between the two words.
    3. If the kept part is implausibly short (< 200 ms or < 40 % of the first part), keep the
       FIRST utterance instead - a single word beats a doubled one.
    Whole clip only when no usable dip exists at all."""
    import numpy as np

    rms, hop = rms_frames(y, sr, 0.010)
    n = len(rms)
    if n < 30:
        return y
    peak = float(rms.max())
    lo, hi = int(0.30 * n), int(0.70 * n)
    below = rms < peak * db_to_lin(-30.0)
    best: tuple[int, int] | None = None
    i = 0
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        if j - i >= 12 and lo <= (i + j) / 2 <= hi and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
        i = j
    if best is not None:
        cut_start, cut_end = best
    else:
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(rms, kernel, mode="same")
        k = lo + int(np.argmin(smooth[lo:hi]))
        if smooth[k] > peak * db_to_lin(-18.0):
            return y
        cut_start = cut_end = k
    second = y[cut_end * hop:]
    first = y[: cut_start * hop]
    voiced = lambda seg: float(np.sum(rms_frames(seg, sr, 0.010)[0] > peak * db_to_lin(-30.0))) * 0.010
    if len(second) < 0.2 * sr or voiced(second) < 0.4 * max(0.05, voiced(first)):
        return first if len(first) >= 0.2 * sr else y
    return second


def postprocess(raw, sr: int, kind: str, word_mode: str):
    """trim -> (double cut) -> quality gate -> pad 40/80 ms -> -16 LUFS -> peak cap.
    -> (float32 clip, None) or (None, reason).

    The peak cap is a plain linear scale-down (spec), so a very peaky short word can end
    up several LU below the target; a limiter would fix that but changes the waveform."""
    import librosa
    import numpy as np
    import pyloudnorm as pyln

    y = np.asarray(raw, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return None, "empty output"
    y, _ = librosa.effects.trim(y, top_db=40)
    if kind == "w" and word_mode == "double":
        y = keep_second_utterance(y, sr)
        y, _ = librosa.effects.trim(y, top_db=40)
    dur = len(y) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    min_s = MIN_WORD_S if kind == "w" else MIN_SENT_S
    if dur < min_s:
        return None, f"too short after trim ({dur:.2f} s < {min_s} s)"
    if dur > MAX_CLIP_S:
        return None, f"too long after trim ({dur:.2f} s > {MAX_CLIP_S} s)"
    if peak <= MIN_PEAK:
        return None, f"peak {peak:.4f} <= {MIN_PEAK}"
    y = np.concatenate([np.zeros(int(sr * PAD_START_S), np.float32), y, np.zeros(int(sr * PAD_END_S), np.float32)])
    measured = y if len(y) >= 0.5 * sr else np.pad(y, (0, sr - len(y)))  # pyloudnorm needs >= 400 ms
    loud = pyln.Meter(sr).integrated_loudness(measured.astype(np.float64))
    if np.isfinite(loud):
        y = y * db_to_lin(TARGET_LUFS - loud)
    peak = float(np.max(np.abs(y)))
    cap = db_to_lin(PEAK_CAP_DBFS)
    if peak > cap:
        y = y * (cap / peak)
    return y.astype(np.float32), None


def atempo_chain(tempo: float) -> str:
    """ffmpeg atempo accepts 0.5-100 per instance; chain instances for stronger stretches."""
    parts = []
    t = tempo
    while t < 0.5:
        parts.append("atempo=0.5")
        t /= 0.5
    while t > 2.0:
        parts.append("atempo=2.0")
        t /= 2.0
    parts.append(f"atempo={t:.4f}")
    return ",".join(parts)


def encode_m4a(ffmpeg: str, y, sr: int, out_path: Path, keep_wav: Path | None = None, tempo: float = 1.0) -> None:
    """16-bit temp WAV -> AAC 48 kbit/s mono 24 kHz, written atomically.

    tempo < 1.0 slows the clip down with ffmpeg's pitch-preserving atempo filter (0.75 = 25 % slower).
    Applied after loudness normalisation; atempo keeps the level."""
    import numpy as np
    import soundfile as sf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_wav is not None:
        keep_wav.parent.mkdir(parents=True, exist_ok=True)
        wav_path = keep_wav
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
    tmp_out = out_path.with_name(out_path.stem + ".tmp.m4a")
    try:
        sf.write(str(wav_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path)]
        if abs(tempo - 1.0) > 1e-3:
            cmd += ["-af", atempo_chain(tempo)]
        cmd += ["-ac", "1", "-ar", str(SR), "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", str(tmp_out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()}")
        os.replace(tmp_out, out_path)
    finally:
        tmp_out.unlink(missing_ok=True)
        if keep_wav is None:
            wav_path.unlink(missing_ok=True)


def m4a_duration(path: Path) -> float | None:
    """Duration from the MP4 'mvhd' box (stdlib only; +faststart puts moov first)."""
    try:
        data = path.read_bytes()
    except OSError:
        return None

    def boxes(buf: bytes, start: int, end: int):
        pos = start
        while pos + 8 <= end:
            size, kind = struct.unpack(">I4s", buf[pos:pos + 8])
            header = 8
            if size == 1:
                if pos + 16 > end:
                    return
                size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
                header = 16
            elif size == 0:
                size = end - pos
            if size < header:
                return
            yield kind, pos + header, min(pos + size, end)
            pos += size

    for kind, body, body_end in boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for sub, sbody, _ in boxes(data, body, body_end):
            if sub != b"mvhd":
                continue
            version = data[sbody]
            if version == 1:
                timescale, duration = struct.unpack(">IQ", data[sbody + 20:sbody + 32])
            else:
                timescale, duration = struct.unpack(">II", data[sbody + 12:sbody + 20])
            return duration / timescale if timescale else None
    return None


# --------------------------------------------------------------------------- reference voice

@dataclass
class Reference:
    source: Path      # what the user supplied
    used: Path        # what the model gets (snippet or the source itself)
    text: str
    seconds: float
    start_s: float
    end_s: float
    sha1: str
    mode: str         # "snippet" | "full"
    text_source: str  # "flag" | "cached" | "whisper"


def resolve_ref_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    return ROOT / p


def transcribe_whisper(y, sr: int, model_name: str = WHISPER_MODEL) -> str:
    """Whisper large-v3 on CUDA (fp16), language=es. Frees the model afterwards."""
    import gc

    import librosa
    import numpy as np
    import torch
    import whisper

    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: CUDA not available — Whisper runs on the CPU (slow)")
    print(f"Transcribing the reference with Whisper {model_name} on {device} ...")
    model = whisper.load_model(model_name, device=device)
    try:
        result = model.transcribe(np.ascontiguousarray(y, dtype=np.float32), language="es",
                                  fp16=(device == "cuda"), verbose=None)
        text = " ".join(str(result.get("text", "")).split())
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return text


def ref_text_from_flag(value: str) -> str:
    p = Path(value)
    text = p.read_text(encoding="utf-8-sig") if p.is_file() else value
    text = " ".join(text.split())
    if not text:
        # F5-TTS would otherwise transcribe the reference itself (transformers Whisper
        # download, language auto-detect) without any visible notice.
        die(f"--ref-text {value!r} is empty")
    return text


def prepare_reference(args, ffmpeg: str | None) -> Reference:
    src = resolve_ref_path(args.ref)
    if not src.exists():
        die(f"Referenzdatei fehlt: {rel(src)} — bitte 60–120 s Sprache eines Porteño ablegen")
    y, sr = load_audio_mono(src, ffmpeg)
    total = len(y) / sr
    if total < 1.0:
        die(f"{rel(src)} is only {total:.2f} s long — the reference needs a few seconds of speech")

    # Use the whole file when it is barely longer than the requested snippet, but never
    # feed F5-TTS more than F5_MAX_REF_S (it would clip the audio and keep the text).
    if total > min(args.ref_seconds + 1.0, F5_MAX_REF_S):
        snip, start_s, end_s = extract_snippet(y, sr, args.ref_seconds)
        import soundfile as sf

        REF_SNIPPET.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(REF_SNIPPET), snip, sr, subtype="PCM_16")
        used, used_y, mode = REF_SNIPPET, snip, "snippet"
    else:
        used, used_y, mode, start_s, end_s = src, y, "full", 0.0, total
    seconds = len(used_y) / sr
    digest = sha1_file(used)

    if args.ref_text:
        text, text_source = ref_text_from_flag(args.ref_text), "flag"
    else:
        text, text_source = "", "whisper"
        if REF_TEXT.exists() and REF_META.exists():
            try:
                meta = json.loads(REF_META.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            same = (meta.get("sha1") == digest and abs(float(meta.get("start_s", -1)) - start_s) < 0.01
                    and abs(float(meta.get("end_s", -1)) - end_s) < 0.01)
            cached = " ".join(REF_TEXT.read_text(encoding="utf-8-sig").split())
            if same and cached:
                text, text_source = cached, "cached"
        if not text:
            text = transcribe_whisper(used_y, sr)
            if not text:
                die("Whisper returned an empty transcript — check that the reference contains speech")
            REF_TEXT.write_text(text + "\n", encoding="utf-8")
            REF_META.write_text(json.dumps({
                "source": rel(src), "used_file": rel(used), "sha1": digest,
                "start_s": round(start_s, 3), "end_s": round(end_s, 3), "duration_s": round(seconds, 3),
                "whisper_model": WHISPER_MODEL, "created": _dt.datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ref = Reference(src, used, text, seconds, start_s, end_s, digest, mode, text_source)
    where = f"{start_s:.1f}–{end_s:.1f} s of {rel(src)}" if mode == "snippet" else "whole file"
    print(f"Reference: {rel(used)} ({seconds:.1f} s, {where}, text: {text_source})")
    print(f'  "{text[:80]}{"…" if len(text) > 80 else ""}"')
    return ref


# --------------------------------------------------------------------------- F5-TTS

class F5Engine:
    """Thin wrapper around f5_tts.api.F5TTS.

    F5TTS.infer() -> infer_process() loads the reference with torchaudio.load, which
    needs torchcodec + FFmpeg shared libraries and fails on this machine. We therefore
    run the same two building blocks infer() uses — preprocess_ref_audio_text() once,
    then infer_batch_process() per item — and load the reference with soundfile.
    Passing a single text chunk also avoids chunk_text() splitting a sentence at
    commas into separately generated, cross-faded pieces.
    """

    def __init__(self, model_spec: str, arch: str | None):
        from f5_tts.api import F5TTS

        # --arch only applies to a checkpoint path: the two named models have a fixed
        # architecture, and the v0/v1 DiT state dicts are shape-compatible, so a wrong
        # config would load without error and produce wrong audio.
        if model_spec == "spanish":
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(SPANISH_REPO, SPANISH_CKPT)
            vocab = hf_hub_download(SPANISH_REPO, SPANISH_VOCAB)
            arch = "F5TTS_Base"  # the Spanish fine-tune uses the original (v0) architecture
        elif model_spec == "base":
            ckpt, vocab = "", ""  # F5TTS downloads SWivid/F5-TTS/F5TTS_v1_Base itself
            arch = "F5TTS_v1_Base"
        else:
            p = Path(model_spec)
            if not p.is_file():
                die(f"--model: {model_spec} is neither 'spanish', 'base' nor an existing checkpoint file")
            ckpt = str(p)
            vocab = str(p.with_name("vocab.txt")) if p.with_name("vocab.txt").is_file() else ""
            arch = arch or "F5TTS_Base"
        self.label = model_spec if model_spec in ("spanish", "base") else p.name
        self.arch = arch
        print(f"Loading F5-TTS {arch} ({self.label}) ...")
        with quiet_stdout():
            self.tts = F5TTS(model=arch, ckpt_file=ckpt, vocab_file=vocab)
        self.ref_audio = None
        self.ref_text = ""
        self.ref_used_seconds = 0.0

    def set_reference(self, ref_path: Path, ref_text: str) -> None:
        """Run F5's reference preprocessing (silence-edge trim, 12 s clip) once and keep
        the result in memory. Called once per process."""
        import librosa
        import numpy as np
        import torch
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        notes: list[str] = []
        with quiet_stdout():
            ref_file, text = preprocess_ref_audio_text(str(ref_path), ref_text, show_info=notes.append)
        try:
            y, sr = load_audio_mono(Path(ref_file))
        finally:
            # F5 writes the processed reference to a temp WAV and never deletes it
            # (it only remembers the path in a process-global cache we do not reuse).
            Path(ref_file).unlink(missing_ok=True)
        if any("clipping" in n for n in notes):
            die(f"F5-TTS clipped the reference to {F5_MAX_REF_S:.0f} s while keeping the full transcript — "
                "the text no longer matches the audio; use a shorter --ref-seconds")
        if sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        self.ref_audio = (torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0), SR)
        self.ref_text = text
        self.ref_used_seconds = len(y) / SR
        print(f"  F5-TTS uses {self.ref_used_seconds:.1f} s of it after trimming the silent edges")

    def generate(self, gen_text: str, seed: int, nfe: int, cfg: float, sway: float, speed: float):
        """-> float32 numpy at 24 kHz, or None when F5 produced nothing."""
        import numpy as np
        from f5_tts.infer.utils_infer import infer_batch_process
        from f5_tts.model.utils import seed_everything

        seed_everything(seed)
        with quiet_stdout():
            wav, sr, _spec = next(infer_batch_process(
                self.ref_audio, self.ref_text, [gen_text], self.tts.ema_model, self.tts.vocoder,
                mel_spec_type=self.tts.mel_spec_type, progress=None, target_rms=0.1, cross_fade_duration=0.15,
                nfe_step=nfe, cfg_strength=cfg, sway_sampling_coef=sway, speed=speed, fix_duration=None,
                device=self.tts.device,
            ))
        if wav is None:
            return None
        if sr != SR:
            raise RuntimeError(f"unexpected sample rate {sr} from F5-TTS")
        return np.asarray(wav, dtype=np.float32).reshape(-1)


@dataclass
class ClipResult:
    ok: bool
    duration: float = 0.0
    attempts: int = 0
    seed: int = 0
    reason: str = ""


def generate_clip(engine: F5Engine, args, ffmpeg: str, kind: str, word_mode: str, gen_text: str,
                  item_id: int, out_path: Path, keep_wav: Path | None) -> ClipResult:
    reason = ""
    for attempt in range(MAX_ATTEMPTS):
        seed = seed_for(args.seed, item_id, attempt)
        try:
            raw = engine.generate(gen_text, seed, args.nfe, args.cfg, args.sway,
                                  effective_speed(kind, gen_text, args.speed, args.word_speed))
        except Exception as exc:  # CUDA OOM, tokenizer trouble, ...
            reason = f"inference error: {type(exc).__name__}: {exc}"
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        if raw is None or raw.size == 0:
            reason = "F5-TTS returned no audio"
            continue
        y, why = postprocess(raw, SR, kind, word_mode)
        if y is None:
            reason = why or "rejected"
            continue
        tempo = args.tempo if kind == "s" else args.word_tempo
        try:
            encode_m4a(ffmpeg, y, SR, out_path, keep_wav, tempo)
        except (RuntimeError, OSError) as exc:  # ffmpeg error, target locked (ffplay), disk full
            # Not seed-dependent, so another attempt would not help.
            return ClipResult(False, attempts=attempt + 1, seed=seed, reason=f"encode error: {exc}")
        return ClipResult(True, round(len(y) / SR / tempo, 2), attempt + 1, seed)
    return ClipResult(False, attempts=MAX_ATTEMPTS, reason=reason)


def prune_failures(names: set[str]) -> None:
    """Drop earlier gen_failures.txt lines for the items about to be (re)generated, so the
    file lists each item at most once and only while its latest attempt failed."""
    if not FAILURES_TXT.exists():
        return
    lines = FAILURES_TXT.read_text(encoding="utf-8-sig").splitlines()
    keep = [ln for ln in lines if ln.split("#", 1)[0].strip() not in names]
    if len(keep) != len(lines):
        FAILURES_TXT.write_text("".join(f"{ln}\n" for ln in keep), encoding="utf-8")


def record_failure(name: str, reason: str) -> None:
    FAILURES_TXT.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_TXT.open("a", encoding="utf-8") as f:
        f.write(f"{name}  # {reason}\n")


def load_engine(args) -> F5Engine:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return F5Engine(args.model, args.arch)


# --------------------------------------------------------------------------- manifest

def write_manifest(reference: dict | None, durations: dict[str, float]) -> dict:
    """Scan app/audio/*.m4a (so manual deletions are reflected); durations come from this
    run, else the previous manifest (only while the file size still matches, i.e. the clip
    was not replaced by a run that died before its next manifest write), else the mvhd box."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    previous: dict = {}
    if MANIFEST.exists():
        try:
            previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    prev = {f.get("name"): f for f in previous.get("files", []) if isinstance(f, dict)}
    entries = []
    for p in AUDIO_DIR.glob("*.m4a"):
        if not re.fullmatch(r"[ws]_\d+\.m4a", p.name):
            continue
        size = p.stat().st_size
        dur = durations.get(p.name)
        if dur is None:
            old = prev.get(p.name)
            if old is not None and old.get("size") == size:
                dur = old.get("dur")
        if dur is None:
            dur = m4a_duration(p) or 0.0
        entries.append({"name": p.name, "dur": round(float(dur), 2), "size": size})
    entries.sort(key=lambda e: e["name"])
    version = hashlib.sha1("\n".join(f"{e['name']}:{e['size']}" for e in entries).encode("utf-8")).hexdigest()[:10]
    payload = {
        "version": version,
        "count": len(entries),
        "bytes": sum(e["size"] for e in entries),
        "reference": reference or previous.get("reference"),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def reference_block(ref: Reference, engine: F5Engine, args) -> dict:
    return {"file": rel(ref.used), "seconds": round(ref.seconds, 2), "mode": args.word_mode, "model": engine.label,
            "source": rel(ref.source), "speed": args.speed, "word_speed": args.word_speed,
            "tempo": args.tempo, "word_tempo": args.word_tempo}


# --------------------------------------------------------------------------- runs

def run_test(args, ffmpeg: str) -> int:
    ref = prepare_reference(args, ffmpeg)
    engine = load_engine(args)
    engine.set_reference(ref.used, ref.text)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    print(f"Test clips -> {TEST_DIR.resolve()}")
    for idx, (stem, kind, mode, text) in enumerate(TEST_ITEMS):
        out = TEST_DIR / f"{stem}.m4a"
        gen_text = gen_text_for(kind, text, mode)
        keep = (TMP_WAV_DIR / f"{stem}.wav") if args.keep_wav else None
        res = generate_clip(engine, args, ffmpeg, kind, mode, gen_text, 9000 + idx, out, keep)
        if res.ok:
            print(f"  {out.resolve()}  {res.duration:.2f} s  (seed {res.seed}, {res.attempts} attempt(s))  «{gen_text}»")
        else:
            failed += 1
            print(f"  {out.resolve()}  FAILED after {res.attempts} attempts: {res.reason}")
    print("Listen and pick a word mode: --word-mode bare (default) or --word-mode double.")
    return 1 if failed else 0


def run_generate(args, ffmpeg: str) -> int:
    t0 = time.time()
    items, notes = load_items()
    for n in notes:
        print(f"note: {n}")
    if not items:
        die("Nothing to generate: data/words.csv and data/sentences.csv are missing or empty "
            "(run tools/validate_data.py first).")
    plan = select_work(args, items)
    for w in plan.warnings:
        print(f"warning: {w}")
    work, skipped = plan.work, plan.skipped
    if not work:
        if plan.selected:
            print("Nothing to do — none of the selected ids can be generated (see the warnings above).")
        else:
            print(f"Nothing to do — {skipped} clip(s) already exist (use --force to regenerate).")
        if args.from_regen:
            update_regen(plan.pending, set())
        manifest = write_manifest(None, {})
        print(f"{rel(MANIFEST)} refreshed: {manifest['count']} files, {manifest['bytes'] / 1e6:.1f} MB")
        return 0
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work)} clip(s) to generate, {skipped} up to date, word mode: {args.word_mode}")

    ref = prepare_reference(args, ffmpeg)
    engine = load_engine(args)
    engine.set_reference(ref.used, ref.text)
    ref_block = reference_block(ref, engine, args)

    from tqdm import tqdm

    generated: dict[str, float] = {}
    failures: list[tuple[Item, str]] = []
    interrupted = False
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    prune_failures({it.name for it in work})
    bar = tqdm(work, unit="clip", dynamic_ncols=True)
    try:
        for n, it in enumerate(bar, start=1):
            bar.set_postfix_str(it.name)
            gen_text = gen_text_for(it.kind, it.text, args.word_mode)
            keep = (TMP_WAV_DIR / f"{it.name}.wav") if args.keep_wav else None
            res = generate_clip(engine, args, ffmpeg, it.kind, args.word_mode, gen_text, it.id, it.out_path, keep)
            if res.ok:
                generated[it.out_path.name] = res.duration
                if res.attempts > 1:
                    tqdm.write(f"{it.name}: ok on attempt {res.attempts} (seed {res.seed})")
            else:
                failures.append((it, res.reason))
                record_failure(it.name, res.reason)
                tqdm.write(f"{it.name}: FAILED — {res.reason}")
            if n % MANIFEST_EVERY == 0:
                write_manifest(ref_block, generated)
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("interrupted — writing manifest for what exists so far")
    finally:
        bar.close()
    manifest = write_manifest(ref_block, generated)

    if args.from_regen:
        # Only ids whose clip was generated leave regen.txt; failed, interrupted,
        # --limit-cut and --only-excluded ids stay for the next run.
        update_regen(plan.pending, {Path(name).stem for name in generated})

    elapsed = time.time() - t0
    print(f"generated {len(generated)} / skipped {skipped} / failed {len(failures)}  |  "
          f"app/audio: {manifest['count']} files, {manifest['bytes'] / 1e6:.1f} MB  |  {elapsed / 60:.1f} min")
    if failures:
        print(f"failed: {', '.join(it.name for it, _ in failures)}  (see {rel(FAILURES_TXT)})")
    if interrupted:
        return 130
    return 1 if failures else 0


def run_dry(args) -> int:
    print("DRY RUN — nothing is loaded or written")
    ffmpeg = find_ffmpeg()
    print(f"ffmpeg: {ffmpeg or 'NOT FOUND'}")
    src = resolve_ref_path(args.ref)
    print(f"reference: {rel(src)} ({'present' if src.exists() else 'MISSING'})")
    if args.test:
        print(f"--test would write to {TEST_DIR.resolve()}:")
        for stem, kind, mode, text in TEST_ITEMS:
            print(f"  {stem}.m4a  «{gen_text_for(kind, text, mode)}»")
        return 0
    items, notes = load_items()
    for n in notes:
        print(f"note: {n}")
    if not items:
        print("Nothing to generate: data/words.csv and data/sentences.csv are missing or empty.")
        return 0
    plan = select_work(args, items)
    for w in plan.warnings:
        print(f"warning: {w}")
    work, skipped = plan.work, plan.skipped
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work)} clip(s) would be generated, {skipped} already exist, word mode: {args.word_mode}, "
          f"model: {args.model}")
    for it in work:
        print(f"  {it.name:<7} -> {rel(it.out_path)}  «{gen_text_for(it.kind, it.text, args.word_mode)}»")
    return 0


# --------------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gen_audio.py",
        description="Generate app/audio/*.m4a from data/words.csv and data/sentences.csv with F5-TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python tools/gen_audio.py --test               comparison clips in data/test_clips/\n"
            "  python tools/gen_audio.py                      generate what is missing\n"
            "  python tools/gen_audio.py --ids w12,s3 --force  redo two items\n"
            "  python tools/gen_audio.py --from-regen         redo everything check_audio.py flagged\n"
        ),
    )
    sel = ap.add_argument_group("selection")
    sel.add_argument("--force", action="store_true", help="regenerate even if the .m4a exists")
    which = sel.add_mutually_exclusive_group()
    which.add_argument("--ids", metavar="LIST", help="only these items: w12,s3,15 (bare number = word id)")
    which.add_argument("--from-regen", action="store_true",
                       help="take ids from data/regen.txt (w_12 / s_3 per line) and regenerate them; "
                            "an id is removed from the file once its clip has been generated")
    sel.add_argument("--only", choices=("words", "sentences"), help="restrict to words or sentences")
    sel.add_argument("--limit", type=int, metavar="N", help="stop after N generated items (smoke tests)")

    voice = ap.add_argument_group("voice / model")
    voice.add_argument("--word-mode", choices=("bare", "double"), default="double",
                       help="single words: 'calle.' (bare) or 'calle. calle.' keeping the second one (double)")
    voice.add_argument("--model", default="spanish", metavar="spanish|base|PATH",
                       help="spanish = jpgallegoar/F5-Spanish (F5TTS_Base arch, default); base = F5TTS_v1_Base; "
                            "or a local .safetensors/.pt (vocab.txt next to it is used when present)")
    voice.add_argument("--arch", choices=("F5TTS_Base", "F5TTS_v1_Base"),
                       help="architecture config for --model PATH only (default: F5TTS_Base)")
    voice.add_argument("--ref", default=str(REF_DEFAULT.relative_to(ROOT).as_posix()), metavar="PATH",
                       help="reference audio (default data/reference.wav)")
    voice.add_argument("--ref-seconds", type=float, default=DEFAULT_REF_SECONDS, metavar="S",
                       help=f"length of the extracted reference snippet, 3-{F5_MAX_REF_S:.0f} "
                            f"(F5-TTS clips longer references internally; default {DEFAULT_REF_SECONDS:g})")
    voice.add_argument("--ref-text", metavar="TEXT_OR_PATH",
                       help="reference transcript (literal text or a file); skips Whisper, does not write data/reference.txt")

    inf = ap.add_argument_group("inference")
    inf.add_argument("--nfe", type=int, default=32, help="ODE steps (default 32)")
    inf.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance strength (default 2.0)")
    inf.add_argument("--speed", type=float, default=0.85,
                     help="speech speed factor (default 1.0); F5-TTS ignores it for texts under 10 bytes "
                          "(forced 0.3), see --word-speed for words of 10-29 bytes")
    inf.add_argument("--word-speed", type=float, default=0.7, metavar="F",
                     help="extra factor on --speed for words whose text is 10-29 UTF-8 bytes ('colectivo.', "
                          "double mode), whose duration budget is otherwise too tight (default 0.7; 1.0 = off)")
    inf.add_argument("--sway", type=float, default=-1.0, help="sway sampling coefficient (default -1.0)")
    inf.add_argument("--tempo", type=float, default=0.75, metavar="F",
                     help="time-stretch factor for SENTENCE clips after synthesis, pitch preserved (ffmpeg atempo); "
                          "0.75 = 25%% slower for beginners. F5-TTS copies the reference speaker's pace, so this is "
                          "the reliable way to slow speech down (default 1.0 = off)")
    inf.add_argument("--word-tempo", type=float, default=1.0, metavar="F",
                     help="same for single-word clips (default 1.0 = off; F5-TTS already speaks short texts slowly)")
    inf.add_argument("--seed", type=int, default=0, metavar="N",
                     help="seed offset; per item seed = N + item id + 1000 * retry (default 0 -> seed = id)")

    mode = ap.add_argument_group("modes")
    mode.add_argument("--test", action="store_true",
                      help="write 4 word clips (bare/double x calle/plata) + 3 sentences to data/test_clips/ and exit")
    mode.add_argument("--dry-run", action="store_true", help="list what would be generated, load nothing")
    mode.add_argument("--keep-wav", action="store_true", help="keep the intermediate 16-bit WAVs in data/tmp_wav/")
    return ap


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if not 3 <= args.ref_seconds <= F5_MAX_REF_S:
        die(f"--ref-seconds must be between 3 and {F5_MAX_REF_S:.0f} (F5-TTS clips longer references)")
    if args.limit is not None and args.limit <= 0:
        die("--limit must be a positive number")
    if args.nfe <= 0:
        die("--nfe must be positive")
    if args.speed <= 0 or args.word_speed <= 0:
        die("--speed and --word-speed must be positive")
    if args.arch and args.model in ("spanish", "base"):
        die("--arch only applies to --model PATH (spanish = F5TTS_Base, base = F5TTS_v1_Base)")
    if args.dry_run:
        return run_dry(args)
    ensure_venv(argv)  # re-executes and exits when we are not inside .venv
    ffmpeg = require_ffmpeg()
    if args.test:
        return run_test(args, ffmpeg)
    return run_generate(args, ffmpeg)


if __name__ == "__main__":
    sys.exit(main())
