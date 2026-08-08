# Models — Product B (official OpenVINO GenAI Kokoro)

Weights are **not** in git. Fetch with `./scripts/fetch_pack.sh`.

## Official pack

| Item | Value |
|------|--------|
| Source | Hugging Face `OpenVINO/kokoro-82M-int8-ov` |
| Local path | `models/kokoro-82M-int8-ov/` |
| `openvino_model.bin` SHA256 | `c879cdd88275b9bfa25e51204d969013d701ea8699e15f99fd1957caf75a29ab` |
| `openvino_model.xml` SHA256 | `a04d5d91e8d6f8d8c1ade28ad331b65827aa148e65515acb4c6725f876257fa5` |
| Voices | `voices/*.bin` (54), e.g. `af_bella`, `af_heart` — shape `(510, 1, 256)` |
| Identity sidecar | `SHIP_PACK_IDENTITY.txt` (optional local record) |

```bash
./scripts/fetch_pack.sh
# or: huggingface-cli download OpenVINO/kokoro-82M-int8-ov --local-dir models/kokoro-82M-int8-ov
```

Env: `KOKORO_GENAI_MODEL=models/kokoro-82M-int8-ov` (default resolves relative to repo root).
