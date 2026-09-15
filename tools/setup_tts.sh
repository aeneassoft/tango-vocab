#!/usr/bin/env bash
# Tango — one-time local TTS environment setup (Windows Git Bash or Linux).
#
# Creates ./.venv, installs PyTorch with a CUDA build that matches the installed
# NVIDIA driver, F5-TTS, Whisper (reference transcription), audio helpers, and
# makes sure ffmpeg is available. Ends with a CUDA self-test that must print True.
#
# Usage: bash tools/setup_tts.sh [--no-models]
#   --no-models   skip pre-downloading the Spanish F5-TTS checkpoint, the vocoder
#                 and Whisper large-v3 (they are fetched on first use instead)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

VENV=".venv"
F5_VERSION="1.1.22"
NO_MODELS=0
for a in "$@"; do case "$a" in --no-models) NO_MODELS=1;; esac; done

log(){  printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn(){ printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }
die(){  printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1) Python
PY=""
for c in python3.12 python3.11 python3.10 python python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
  case "$v" in 3.10|3.11|3.12) PY="$c"; break;; esac
done
[ -n "$PY" ] || die "Python 3.10-3.12 not found on PATH"
log "Python: $("$PY" --version) ($(command -v "$PY"))"

# ---------------------------------------------------------------- 2) venv
if [ ! -f "$VENV/pyvenv.cfg" ]; then
  log "Creating virtualenv $VENV"
  "$PY" -m venv "$VENV" || die "venv creation failed"
fi
if [ -x "$VENV/Scripts/python.exe" ]; then VPY="$VENV/Scripts/python.exe"; else VPY="$VENV/bin/python"; fi
"$VPY" -m pip install -q --upgrade pip wheel setuptools || die "pip upgrade failed"

# ---------------------------------------------------------------- 3) CUDA driver -> wheel index
DRV=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [ -z "$DRV" ]; then warn "nvidia-smi not found; assuming a CUDA 12.8 capable driver"; DRV="12.8"; fi
MAJ=${DRV%%.*}; MIN=${DRV##*.}
log "Driver reports CUDA $DRV"
if   [ "$MAJ" -ge 13 ]; then CANDS="cu130 cu128 cu126"
elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then CANDS="cu128 cu126"
elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then CANDS="cu126 cu124"
else CANDS="cu121"; fi

cuda_ok(){ "$VPY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }

# ---------------------------------------------------------------- 4) torch
if cuda_ok; then
  log "torch with CUDA already present: $("$VPY" -c 'import torch;print(torch.__version__)')"
else
  for idx in $CANDS; do
    log "Installing torch + torchaudio from $idx"
    if "$VPY" -m pip install --upgrade torch torchaudio --index-url "https://download.pytorch.org/whl/$idx"; then
      if cuda_ok; then log "CUDA works with $idx"; break; else warn "$idx installed but CUDA unavailable, trying next"; fi
    fi
  done
  cuda_ok || die "No CUDA-enabled torch could be installed. Update the NVIDIA driver, then re-run."
fi
TORCH_IDX=$("$VPY" -c 'import torch;print(torch.__version__.split("+")[-1])')

# ---------------------------------------------------------------- 5) F5-TTS + helpers
log "Installing F5-TTS $F5_VERSION, Whisper, audio helpers"
"$VPY" -m pip install "f5-tts==$F5_VERSION" openai-whisper pyloudnorm soundfile librosa huggingface_hub \
  || die "pip install of F5-TTS / helpers failed"

# f5-tts pulls its own torch requirement; if it replaced the CUDA build, put it back.
if ! cuda_ok; then
  warn "CUDA lost after installing f5-tts; reinstalling torch from $TORCH_IDX"
  "$VPY" -m pip install --upgrade --force-reinstall --no-deps torch torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_IDX"
  cuda_ok || die "CUDA still unavailable after reinstalling torch"
fi

# ---------------------------------------------------------------- 6) ffmpeg
find_ffmpeg(){
  command -v ffmpeg 2>/dev/null && return 0
  local la; la=$(cygpath -u "${LOCALAPPDATA:-}" 2>/dev/null || echo "$HOME/AppData/Local")
  local p
  for p in "$la"/Microsoft/WinGet/Links/ffmpeg.exe "$la"/Microsoft/WinGet/Packages/*FFmpeg*/*/bin/ffmpeg.exe /c/ffmpeg/bin/ffmpeg.exe /usr/local/bin/ffmpeg; do
    [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
FF=$(find_ffmpeg || true)
if [ -z "$FF" ] && command -v winget >/dev/null 2>&1; then
  log "ffmpeg missing — installing via winget (Gyan.FFmpeg)"
  winget install --id Gyan.FFmpeg -e --silent --accept-package-agreements --accept-source-agreements || warn "winget install failed"
  FF=$(find_ffmpeg || true)
fi
[ -n "$FF" ] || die "ffmpeg not found. Install it (Windows: winget install Gyan.FFmpeg; Linux: apt install ffmpeg) and re-run."
log "ffmpeg: $FF"

# ---------------------------------------------------------------- 7) models (optional)
if [ "$NO_MODELS" -eq 0 ]; then
  log "Pre-downloading Spanish F5-TTS checkpoint (jpgallegoar/F5-Spanish) and vocoder"
  "$VPY" - <<'PY' || warn "model pre-download failed (will retry on first generation)"
from huggingface_hub import hf_hub_download
for f in ("model_1200000.safetensors", "vocab.txt"):
    print(hf_hub_download("jpgallegoar/F5-Spanish", f))
try:
    from f5_tts.infer.utils_infer import load_vocoder
    load_vocoder(vocoder_name="vocos", is_local=False, device="cpu")
    print("vocoder ok")
except Exception as e:
    print("vocoder pre-download skipped:", e)
PY
  log "Pre-downloading Whisper large-v3 (~3 GB, only needed once for the reference transcript)"
  "$VPY" -c 'import whisper; whisper.load_model("large-v3", device="cpu"); print("whisper ok")' || warn "whisper pre-download failed"
fi

# ---------------------------------------------------------------- 8) self-test
log "Self-test"
"$VPY" - <<'PY'
import torch, importlib
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| vram GB:", round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))
for m in ("f5_tts", "whisper", "pyloudnorm", "soundfile", "librosa"):
    importlib.import_module(m); print("import ok:", m)
print(torch.cuda.is_available())
PY
cuda_ok || die "torch.cuda.is_available() is False"
log "Done. Next: python tools/gen_audio.py --test   (needs data/reference.wav)"
