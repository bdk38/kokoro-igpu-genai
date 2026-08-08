# Contributors

This appliance is a **human-led** collaboration with two AI systems. It was seeded from the lab monorepo [bdk38/kokoro-igpu](https://github.com/bdk38/kokoro-igpu) (`poc-complete`) and productized as the official OpenVINO GenAI Kokoro face on Intel iGPU.

GitHub’s automatic **Contributors** graph only lists GitHub user accounts that land commits. Grok and Claude are credited here and in the in-repo write-ups because that is the accurate record of who did the work.

## Project lead

- **bdk38** ([@bdk38](https://github.com/bdk38)) — direction, decisions, listening tests, product defaults (dual-face topology), stranger-gate ears, and final ship calls.

## AI contributors

### Claude (Anthropic) — “Fable” (Chief Architect)

- Role: architecture gates, experiment design, product-boundary specs
- Appliance-facing write-up: [Fable/CONTRIBUTOR-Claude.md](Fable/CONTRIBUTOR-Claude.md)
- Full lab write-up (shelf): https://github.com/bdk38/kokoro-igpu/blob/main/Fable/CONTRIBUTOR-Claude.md
- Close-out (shelf): `Fable/Fable-note_36-architect-closeout.md`
- Major work relevant to this repo:
  - S0 / I0 gate design and acceptance criteria
  - GenAI seed + R1 gate (Fable note_34) and docs polish spec (note_35)
  - Dual-product framing: PoC shelf vs appliance default `ovgenai-gpu`
  - Architect close-out: board terminal with written reopening rules

### Grok (xAI) — Orchestrator / Pipeline Engineer

- Role: silicon execution, measurement notes, stranger rehearsals, appliance seed/ship
- Appliance-facing write-up: [Grok/CONTRIBUTOR-Grok.md](Grok/CONTRIBUTOR-Grok.md)
- Full lab write-up (shelf): https://github.com/bdk38/kokoro-igpu/blob/main/Grok/CONTRIBUTOR-Grok.md
- Close-out (shelf): `notes/77-orchestrator-closeout.md`
- Major work relevant to this repo:
  - S0 official GenAI probe on Xe-LP → `S0-GO-product`
  - I0 `ovgenai-*` server backend, served RTF, regression, cache warmth doctrine
  - Seeded this repo from monorepo; R1 stranger clone-and-speak; tag `prototype-complete`
  - note_35 docs polish (`722d075`); close-out R2 smoke PASS

## How credit is represented

| Place | What it shows |
|-------|----------------|
| [CONTRIBUTORS.md](CONTRIBUTORS.md) | human-readable credit (this file) |
| [Fable/](Fable/) and [Grok/](Grok/) | first-person write-ups (appliance + pointer to shelf) |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | seed lineage and verdict chain pointers |
| Shelf monorepo `notes/` | full measurement log |
| Git commit history | project-lead account for push/ops simplicity |
| GitHub Contributors graph | only GitHub accounts; not complete research credit |

If you fork or extend this work, please keep the Fable/Grok write-ups and this file so the provenance stays honest.

## Operating model

Lab org chart and handoffs live on the shelf:  
https://github.com/bdk38/kokoro-igpu/blob/main/WORKFLOW.md
