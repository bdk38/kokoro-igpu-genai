#!/usr/bin/env bash
# Product B smoke: ovgenai-cpu required; ovgenai-gpu if OpenVINO sees GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/venv/bin/python"
[[ -x "$PY" ]] || PY="${PYTHON:-python3}"
OUT="${ROOT}/artifacts/smoke"
mkdir -p "$OUT"
PORT="${SMOKE_PORT:-8890}"
FOX='The quick brown fox jumps over the lazy dog.'
PACK="${KOKORO_GENAI_MODEL:-$ROOT/models/kokoro-82M-int8-ov}"

echo "[smoke] root=$ROOT pack=$PACK"
if [[ ! -f "$PACK/openvino_model.xml" ]]; then
  echo "[smoke] pack missing — run ./scripts/fetch_pack.sh first" >&2
  exit 2
fi

fail=0
run_one() {
  local backend="$1"
  local tag="$2"
  local log="$OUT/${tag}.log"
  if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "[smoke] port $PORT busy"; return 1
  fi
  env KOKORO_BACKEND="$backend" KOKORO_TTS_CACHE=0 \
    KOKORO_GENAI_MODEL="$PACK" \
    "$PY" scripts/kokoro_server.py --host 127.0.0.1 --port "$PORT" \
    >"$log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 1 180); do
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[smoke] $tag died"; tail -40 "$log"; return 1
    fi
    sleep 1
  done
  [[ "$ok" == 1 ]] || { echo "[smoke] $tag health timeout"; tail -40 "$log"; kill "$pid" 2>/dev/null || true; return 1; }
  local health
  health=$(curl -sf "http://127.0.0.1:${PORT}/health")
  echo "[smoke] $tag health=$health"
  "$PY" - <<PY
import json, urllib.request
body=json.dumps({"input":"$FOX","voice":"af_bella","response_format":"wav"}).encode()
req=urllib.request.Request("http://127.0.0.1:${PORT}/v1/audio/speech", data=body,
  headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=900) as r:
    open("$OUT/${tag}.wav","wb").write(r.read())
print("wrote $OUT/${tag}.wav")
PY
  local sz
  sz=$(wc -c <"$OUT/${tag}.wav" | tr -d ' ')
  echo "[smoke] $tag bytes=$sz path=$OUT/${tag}.wav"
  [[ "$sz" -ge 10000 ]] || { fail=1; echo "[smoke] $tag too small"; }
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 1
}

echo "=== ovgenai-cpu (required) ==="
run_one ovgenai-cpu ovgenai_cpu || fail=1

if "$PY" -c "import openvino as ov; raise SystemExit(0 if 'GPU' in ov.Core().available_devices else 1)" 2>/dev/null; then
  echo "=== ovgenai-gpu ==="
  run_one ovgenai-gpu ovgenai_gpu || fail=1
  if command -v intel_gpu_top >/dev/null 2>&1; then
    echo "[smoke] tip: intel_gpu_top during GPU synth for RCS fingerprint"
  fi
else
  echo "[smoke] no GPU device — skipped ovgenai-gpu"
fi

ls -la "$OUT"/*.wav 2>/dev/null || true
if [[ "$fail" -ne 0 ]]; then echo "[smoke] FAIL"; exit 1; fi
echo "[smoke] PASS"; exit 0
