# Holster Memory

> *Slow memory holsters it. VRAM draws the gun and pulls the trigger.*

**Holster Memory** is a tiered GPU memory architecture in which a large, slower memory pool holds the full workload while VRAM functions as a high-speed execution cache — staging only the data needed at the moment of computation.

**Prior art established:** December 2025  
**Author:** James Yarber  
**Status:** Pre-hardware prototype — harness and architecture complete, execution benchmarks pending hardware availability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20649203.svg)](https://doi.org/10.5281/zenodo.20649203)


---

## The Problem

Consumer GPUs have a hard VRAM wall. A model that needs 28GB on a 24GB card simply fails. Current workarounds either require expensive hardware upgrades or accept that the workload won't run at all.

## The Idea

VRAM should be treated as a cache, not a container.

Instead of:
> "If it doesn't fit in VRAM, it cannot run."

Holster Memory makes it:
> "If it doesn't fully fit in VRAM, it can still run by staging the right pieces into VRAM at the right time."

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Holster Runtime                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Scheduler  │→ │ Transfer Mgr │←→│ VRAM Pool Mgr  │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│         ↓                ↓                   ↓          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Layer Graph │  │ Async Streams│  │ Active/Hot Set │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
         ↓                    ↓                   ↓
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Passive RAM │    │ Compute Engine  │    │ Cold Storage    │
│ (Holster)   │    │ (GPU + Streams) │    │ (Disk/SSD)      │
└─────────────┘    └─────────────────┘    └─────────────────┘
```

**Three tiers:**
- **Active VRAM** — executing now
- **Passive RAM (The Holster)** — full model, staged and warm
- **Cold Storage** — disk, last resort

**Four components:**
- **Scheduler** — predicts next N layers, decides eviction candidates
- **Transfer Manager** — async copies via CUDA streams, pinned memory
- **VRAM Pool Manager** — enforces hard VRAM budget, prevents fragmentation
- **Metrics Collector** — measures overlap efficiency, stall frequency, hit rate

## Three Implementation Paths

| Path | Description | Hardware Required |
|------|-------------|-------------------|
| 1 | Software-only, system RAM as passive tier | Existing PC |
| 2 | Pinned buffers + async prefetch + explicit hot/cold | Existing PC |
| 3 | PCIe/CXL-attached expansion memory as passive tier | New hardware |

Paths 1 and 2 validate the concept. Path 3 is a future optimization.

## Design Principles

1. **The GPU should never be surprised** — if data is missing at execution time, the user feels the stall
2. **VRAM is a cache, not a warehouse** — hold what is hot, not everything
3. **Promotion matters more than raw capacity** — right data at the right time beats bigger memory
4. **Graceful degradation beats hard failure** — slower is better than OOM
5. **Predictability beats guesswork** — explicit, measurable promotion and eviction rules

## Why AI Workloads Are a Strong Fit

- Layer-by-layer execution — access order is known in advance
- Weight reuse — same tensors accessed repeatedly
- Predictable structure — prefetching is highly effective
- Chunkable computation — natural staging boundaries

## Getting Started

```bash
pip install torch transformers accelerate
python harness/benchmark.py --profile profiles/am4_ddr4.json --model 3b
```

See `docs/` for full architecture documentation and `benchmarks/` for expected results per hardware profile.

## Repository Structure

```
holster-memory/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md          # Full conceptual spec
│   ├── PRIOR_ART.md             # Timestamp and defensive publication
│   └── PLATFORM_COMPARISON.md  # DDR4 vs DDR5, AM4 vs AM5 bandwidth analysis
├── harness/
│   ├── holster.py               # Core HolsterModel implementation
│   ├── scheduler.py             # LayerScheduler
│   ├── metrics.py               # HolsterMetrics
│   └── benchmark.py             # CLI benchmark runner
├── profiles/
│   ├── am4_ddr4.json            # AM4 + DDR4 hardware profile
│   └── am5_ddr5.json            # AM5 + DDR5 hardware profile
└── benchmarks/
    └── results/                 # Populated when hardware is available
```

## Prior Art

This concept was developed and documented in December 2025 as a defensive publication to establish prior art. See `docs/PRIOR_ART.md` for the full record.

The novelty is not inventing memory tiers from nothing — CPU cache hierarchies, texture streaming, and ZeRO-style offloading all exist. The novelty is applying a clear, intentional tiered model to consumer GPU workloads and framing VRAM explicitly as an active execution cache rather than a fixed all-or-nothing container.

---

*Holster Memory keeps the workload loaded in a slower passive tier so VRAM can focus on executing the shot.*


---

## Documents

- [docs/CONCEPT.md](docs/CONCEPT.md) — full long-form concept and implementation-path document
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — harness architecture
- [docs/PRIOR_ART.md](docs/PRIOR_ART.md) — prior art positioning
- [docs/PLATFORM_COMPARISON.md](docs/PLATFORM_COMPARISON.md)

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).

This repository is a defensive publication. The concept was developed and named in December 2025; publication establishes a public, timestamped record so the architecture remains open for anyone to implement.
