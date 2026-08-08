# kokoro-igpu-genai

**Official Kokoro-82M int8 (OpenVINO GenAI)** served on a budget Intel iGPU (Xe-LP / Alder Lake class).

| | |
|--|--|
| **Default** | `ovgenai-gpu` |
| **No-GPU fallback** | `KOKORO_BACKEND=ovgenai-cpu` |
| **Warm steady RTF** (validation host) | ~**0.73** fox / ~**0.72** multi |
| **Version** | **2.0.0** |

**Honest tax (same breath):** the **first** synthesis of a **novel** text length can take **tens of seconds** (shape-keyed GPU JIT). Repeats and warmed shapes are realtime-class. Do **not** quote steady RTF as first-utterance latency.

Lab story and PoC (ONNX-era) live on the shelf:  
https://github.com/bdk38/kokoro-igpu (`poc-complete`) — see [docs/PROVENANCE.md](docs/PROVENANCE.md).

---

## Run it

### Install

```bash
git clone https://github.com/bdk38/kokoro-igpu-genai.git
cd kokoro-igpu-genai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# system: espeak-ng (GenAI G2P fallback), e.g. sudo apt install espeak-ng
./scripts/fetch_pack.sh
```

### Start (default = iGPU GenAI)

```bash
python scripts/kokoro_server.py --host 0.0.0.0 --port 8880
# KOKORO_BACKEND=ovgenai-gpu
```

No GPU?

```bash
KOKORO_BACKEND=ovgenai-cpu python scripts/kokoro_server.py --host 0.0.0.0 --port 8880
```

### Hear it

```bash
curl -sS -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"The quick brown fox jumps over the lazy dog.","voice":"af_bella","response_format":"wav"}' \
  -o fox.wav
```

### Smoke

```bash
./scripts/smoke.sh
# artifacts/smoke/ovgenai_cpu.wav required
# artifacts/smoke/ovgenai_gpu.wav if GPU present
```

Deploy tip: `KOKORO_TTS_CACHE=1` and chunk-shaped `KOKORO_WARM_TEXT='phrase one|phrase two'`.

---

## Configuration

| Variable | Default | Notes |
|----------|---------|--------|
| `KOKORO_BACKEND` | `ovgenai-gpu` | `ovgenai-cpu` fallback |
| `KOKORO_GENAI_MODEL` | `models/kokoro-82M-int8-ov` | pack directory |
| `KOKORO_DEFAULT_VOICE` | `af_bella` | `af_heart` first-class |
| `KOKORO_TTS_CACHE` | `0` | set `1` for repeats |
| `KOKORO_TTS_CACHE_DIR` | `<repo>/cache/tts` | |
| `KOKORO_WARM_TEXT` | empty | `\|`-separated **chunk-shaped** pins |

Pack hashes: [MODELS.md](MODELS.md).

---

## Performance honesty

| Mode | Expect |
|------|--------|
| Warm / repeat shape | RTF ~0.7 class on validation Xe-LP |
| First novel length | multi-second to ~tens of seconds extra |
| Mitigation | TTS cache + `KOKORO_WARM_TEXT` |

---

## Voices

54 embeddings in the official pack. Default **af_bella**; **af_heart** excellent alternate.  
Blends not supported on GenAI path. Timbre differs from older v0.19 ONNX bella.

---

## Architecture

```text
text → server chunker → per-chunk GenAI generate() → PCM assemble
                      ↘ optional C1/C2 disk cache (unit = chunk/request)
```

---

## Provenance

[docs/PROVENANCE.md](docs/PROVENANCE.md) → monorepo shelf for S0/I0 notes and the ONNX PoC.

---

## Limits

- Novel-shape first-hit tax  
- No GenAI voice blends  
- Single-process server  
- Weights downloaded (not in git)  
- ONNX backends may exist in the binary for lineage; this product face is GenAI-only  

---

## Credits

See [CONTRIBUTORS.md](CONTRIBUTORS.md), [LICENSE](LICENSE).  
Kokoro / OpenVINO: upstream projects. Seeded from bdk38/kokoro-igpu.
