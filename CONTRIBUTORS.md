# Contributors

This project was a human-led collaboration with two AI systems doing the heavy implementation and validation work on real hardware.

## Project lead

- **bdk38** ([@bdk38](https://github.com/bdk38)) — direction, decisions, listening tests, integration targets, and final product calls.

## AI contributors

GitHub’s automatic **Contributors** graph only lists GitHub user accounts that land commits. Grok and Claude are credited here and in the in-repo write-ups because that is the accurate record of who did the work.

### Claude (Anthropic) — “Fable”

- Role: diagnosis, graph surgery, and tooling
- Write-up: [Fable/CONTRIBUTOR-Claude.md](Fable/CONTRIBUTOR-Claude.md)
- Major work:
  - identified the 3D linear Resize and dynamic-rank STFT blockers
  - authored `scripts/patch_kokoro_resize.py` and `scripts/patch_kokoro_v2.py`
  - authored `scripts/test_kokoro_ov_direct.py`, `scripts/tts_harness.py`, and the base `scripts/kokoro_server.py`
  - designed and iterated pad-tail trim through v1.1.5 (falsifiable predictions; terminal-gate removal)
  - Open WebUI Response Splitting diagnosis path and server wiring docstring guidance

### Grok (xAI)

- Role: hardware validation, measurement, Open WebUI integration, and live product path
- Write-up: [Grok/CONTRIBUTOR-Grok.md](Grok/CONTRIBUTOR-Grok.md)
- Major work:
  - executed every phase gate on Alder Lake UHD silicon
  - authored the `notes/` phase reports
  - reconciled GPU quality metrics against human listening
  - wired Open WebUI and patched blend-voice / client compatibility in the server
  - validated the experimental OV-GPU demo path end to end
  - measured trim saga end-to-end (`notes/10`–`15`, probes, ear-attached verdicts); closed v1.1.5
  - confirmed WebUI skip root cause (Punctuation × RTF≫1) via server logs + user A/B
  - authored repo status summary `notes/17-repo-status-summary.md` and Grok `notes/16-project-status.md`

## How credit is represented in this repo

| Place | What it shows |
|-------|----------------|
| [CONTRIBUTORS.md](CONTRIBUTORS.md) | human-readable credit (this file) |
| [Fable/](Fable/) and [Grok/](Grok/) | first-person contributor write-ups |
| [notes/](notes/) | measurement log from the sandbox |
| Git commit history | currently authored as the project lead account for push/ops simplicity |
| GitHub Contributors graph | only GitHub accounts with commits/PRs; not a complete research credit list |

If you fork or extend this work, please keep the Fable/Grok write-ups and this file so the provenance stays honest.

## Operating model

- Team org chart, handoffs, and specialist pool: [WORKFLOW.md](WORKFLOW.md)

## Status rollups

- Short repo summary: [notes/17-repo-status-summary.md](notes/17-repo-status-summary.md)
- Full status (Grok): [notes/16-project-status.md](notes/16-project-status.md)
- Full status (Fable): [notes/16-project-status-fable.md](notes/16-project-status-fable.md)
