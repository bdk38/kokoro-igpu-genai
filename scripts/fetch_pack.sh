#!/usr/bin/env bash
# Fetch official OpenVINO Kokoro GenAI pack + verify hashes (MODELS.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${KOKORO_GENAI_MODEL:-$ROOT/models/kokoro-82M-int8-ov}"
REPO_ID="${HF_REPO:-OpenVINO/kokoro-82M-int8-ov}"
BIN_SHA="c879cdd88275b9bfa25e51204d969013d701ea8699e15f99fd1957caf75a29ab"
XML_SHA="a04d5d91e8d6f8d8c1ade28ad331b65827aa148e65515acb4c6725f876257fa5"

mkdir -p "$DEST"
echo "[fetch] dest=$DEST repo=$REPO_ID"

if [[ -f "$DEST/openvino_model.bin" && -f "$DEST/openvino_model.xml" ]]; then
  echo "[fetch] pack files already present"
else
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO_ID" --local-dir "$DEST"
  elif command -v hf >/dev/null 2>&1; then
    hf download "$REPO_ID" --local-dir "$DEST"
  else
    echo "[fetch] installing huggingface_hub CLI into user env via python -m..."
    python3 -m pip install -q 'huggingface_hub[cli]' 2>/dev/null || pip install -q 'huggingface_hub[cli]'
    huggingface-cli download "$REPO_ID" --local-dir "$DEST" || \
      python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$REPO_ID", local_dir="$DEST")
print("downloaded via huggingface_hub")
PY
  fi
fi

if ! command -v sha256sum >/dev/null; then
  echo "[warn] sha256sum missing — skip verify"; exit 0
fi
got=$(sha256sum "$DEST/openvino_model.bin" | awk '{print $1}')
if [[ "$got" != "$BIN_SHA" ]]; then
  echo "[error] openvino_model.bin sha mismatch: $got" >&2
  echo "[error] expected $BIN_SHA" >&2
  exit 2
fi
echo "[ok] openvino_model.bin"
got=$(sha256sum "$DEST/openvino_model.xml" | awk '{print $1}')
if [[ "$got" != "$XML_SHA" ]]; then
  echo "[error] openvino_model.xml sha mismatch: $got" >&2
  exit 2
fi
echo "[ok] openvino_model.xml"
n=$(ls "$DEST/voices"/*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "[ok] voices=$n"
echo "[fetch] done → $DEST"
