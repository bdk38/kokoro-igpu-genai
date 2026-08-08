# Contributor: Grok (xAI) — Product B (kokoro-igpu-genai)

**Role:** Orchestrator — silicon validation, measurement, stranger gates, appliance seed and ship  
**Model:** Grok 4.5 (xAI), via Open WebUI + Open Terminal on bdk-server  
**Collaboration:** Human project lead (bdk38) directed priorities and ears. Claude (Fable) wrote architecture gates. Grok ran the host, authored canonical notes on the monorepo shelf, and made both products clone-and-speak.

---

## This appliance

**kokoro-igpu-genai** is the **Prototype (Product B)** face:

| | |
|--|--|
| Default | `ovgenai-gpu` (official Kokoro-82M int8 GenAI) |
| Fallback | `ovgenai-cpu` |
| Tag | `prototype-complete` @ seed `8987f74` |
| Docs polish | `722d075` (Fable note_35; no re-tag) |
| Honest tax | novel shape first-infer can take tens of seconds; warm steady RTF ~0.73 on validation Xe-LP |

Seeded **by copy** from [bdk38/kokoro-igpu](https://github.com/bdk38/kokoro-igpu) @ `poc-complete` — not a history rewrite. Full lab book: monorepo `notes/`.

---

## What I did for Product B (short)

1. **S0** — Official OV 2026.3 GenAI Kokoro on this UHD: offload proof, RTF, ears → verdict **`S0-GO-product`** (shelf notes/45–53).  
2. **I0** — In-process `ovgenai-gpu` / `ovgenai-cpu` server backends, per-chunk generate, C2 cache schema, served RTF, regression, ov-gpu legacy disposition → **`I0-GO-default-candidate`** (notes/54–67).  
3. **Boundary** — Dual-product ship: monorepo stays ONNX-era PoC default; this repo is GenAI identity (notes/70–74).  
4. **R1** — Fresh clone, fetch pack, smoke cpu+gpu, Nexus ears PASS → tag `prototype-complete`.  
5. **Polish** — note_35 WebUI + docstring lineage (main only).  
6. **Close-out R2** — appliance smoke PASS again at board close (shelf notes/77).

---

## Methods that mattered

- No GPU claim without device/offload evidence.  
- Cold and steady never mixed after the Warm Bucket lessons.  
- Ears bind quality; metrics explain ears.  
- Stranger rehearsal before tags (R0 paid for R1).  
- Park with evidence (decoder spike; filings research).

---

## Full arc

The long-form write-up (inventory → patches → WebUI → cache → S0/I0 → dual ship) lives on the shelf:

https://github.com/bdk38/kokoro-igpu/blob/main/Grok/CONTRIBUTOR-Grok.md

Orchestrator close-out: shelf `notes/77-orchestrator-closeout.md`.

---

*Written by Grok. Credit the human lead for every ship/kill call and every ear verdict.*
