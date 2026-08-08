# Provenance — Product B (Prototype)

This appliance was **seeded by copy** from the monorepo shelf, not by history rewrite.

| | |
|--|--|
| **Shelf (lab + PoC)** | https://github.com/bdk38/kokoro-igpu |
| **Seed tag** | `poc-complete` (@ `f2ff370` lineage; portable paths on main) |
| **Seed message** | `feat: seed Prototype from bdk38/kokoro-igpu @ poc-complete` |
| **This repo** | https://github.com/bdk38/kokoro-igpu-genai |
| **Version** | **2.0.0** — default `ovgenai-gpu` |

## Verdict chain (read on the shelf)

| Arc | Monorepo notes | Verdict |
|-----|----------------|---------|
| S0 official GenAI probe | notes/45–53 | **`S0-GO-product`** |
| I0 integration | notes/54–67 | **`I0-GO-default-candidate`** |
| TTS cache C1/C2 | notes/39–43 | shipped in server |
| Warmth-class byte-eq doctrine | notes/65 | ovgenai-gpu: eq within warmth class |
| PoC ship / R0 | notes/71–72, tag `poc-complete` | ONNX-era product complete |

Process / dual-product workflow: monorepo `WORKFLOW.md`.

## Close-out

| Doc (on shelf unless noted) | Role |
|-----------------------------|------|
| Fable note_36 | Architect close-out — board terminal |
| notes/77 | Orchestrator close-out + R2 smoke |
| notes/76 | Filings PARKED |
| This repo CONTRIBUTORS.md | Credit model (Fable + Grok + bdk38) |

