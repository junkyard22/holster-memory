# Platform Comparison: AM4/DDR4 vs AM5/DDR5

## Why This Document Exists

Holster Memory's passive tier is system RAM. The bandwidth between RAM and VRAM — mediated by the PCIe bus — is the primary performance variable in the system. Different platform choices produce meaningfully different bandwidth envelopes, which affects prefetch window sizing and expected tokens/sec.

This document captures the expected characteristics of both platforms so benchmark results can be interpreted correctly regardless of which hardware ends up being used.

---

## Bandwidth Hierarchy

```
VRAM (on-card)
    ~900 GB/s (RTX 3090 internal)
         ↕
    PCIe Bus
         ↕
System RAM
    ~40-90 GB/s (platform dependent)
         ↕
    Storage (SSD)
    ~3-7 GB/s (NVMe)
```

The PCIe bus is the critical bottleneck. Everything in Holster Memory's scheduler design exists to hide this latency.

---

## Platform Profiles

### AM4 + DDR4

| Spec | Value |
|------|-------|
| RAM type | DDR4 |
| Typical speed | 3200–3600 MHz |
| Theoretical RAM bandwidth | ~50 GB/s (dual channel) |
| PCIe version | 4.0 (Ryzen 5000) |
| PCIe x16 bandwidth | ~32 GB/s bidirectional |
| PCIe x16 practical (one direction) | ~28–30 GB/s |
| Platform longevity | End of life — no new CPUs |
| DDR4 pricing | Mature, low cost |

**Holster Memory implications:**
- PCIe 4.0 x16 gives ~28 GB/s practical transfer to GPU
- A 7B model (~14GB fp16) can be fully transferred in ~0.5s
- Layer-by-layer transfer time per layer (~400MB): ~14ms
- Prefetch window of 2–3 layers provides adequate overlap at most compute speeds
- Strong value proposition: cheap to build, proves the concept fully

---

### AM5 + DDR5

| Spec | Value |
|------|-------|
| RAM type | DDR5 |
| Typical speed | 5600–6400 MHz |
| Theoretical RAM bandwidth | ~80–90 GB/s (dual channel) |
| PCIe version | 5.0 |
| PCIe x16 bandwidth | ~64 GB/s bidirectional |
| PCIe x16 practical (one direction) | ~56–60 GB/s |
| Platform longevity | Current — ongoing support |
| DDR5 pricing | Still elevated vs DDR4 |

**Holster Memory implications:**
- PCIe 5.0 x16 gives ~56 GB/s practical transfer to GPU
- A 7B model (~14GB fp16) can be fully transferred in ~0.25s
- Layer-by-layer transfer time per layer (~400MB): ~7ms
- Prefetch window of 1–2 layers may be sufficient at higher compute speeds
- Higher bandwidth reduces scheduler pressure — more forgiving to tune
- Better long-term platform for eventual distillation workloads

---

## Expected Performance Envelope

These are theoretical estimates. Actual results depend on layer size, model architecture, and prefetch window tuning. Populate with real numbers once hardware is available.

| Metric | AM4/DDR4 Estimate | AM5/DDR5 Estimate | Baseline (CPU offload) |
|--------|-------------------|-------------------|------------------------|
| Tokens/sec (7B, fp16) | TBD | TBD | ~5–8 |
| Max model size (24GB VRAM) | RAM-limited | RAM-limited | RAM-limited |
| OOM rate | 0% (target) | 0% (target) | 0% |
| Stalls per generation | TBD | TBD | N/A |
| Effective PCIe util | TBD | TBD | N/A |
| Prefetch window (optimal) | ~3 layers | ~2 layers | N/A |

---

## Scheduler Tuning by Platform

The prefetch window is the primary tunable parameter. A wider window reduces stalls but increases VRAM pressure. The optimal window is the smallest one that keeps the GPU fully fed.

### AM4/DDR4 Starting Config

```python
PLATFORM_CONFIG = {
    "platform": "AM4",
    "ram_type": "DDR4",
    "ram_speed_mhz": 3600,
    "pcie_gen": 4,
    "pcie_lanes": 16,
    "pcie_practical_gbps": 28.0,
    "vram_gb": 24,
    "gpu": "RTX 3090",
    "recommended_prefetch_window": 3,
    "recommended_vram_budget_gb": 20,  # Leave headroom for KV cache
    "notes": "Wider prefetch window compensates for lower PCIe bandwidth"
}
```

### AM5/DDR5 Starting Config

```python
PLATFORM_CONFIG = {
    "platform": "AM5",
    "ram_type": "DDR5",
    "ram_speed_mhz": 6000,
    "pcie_gen": 5,
    "pcie_lanes": 16,
    "pcie_practical_gbps": 56.0,
    "vram_gb": 24,
    "gpu": "RTX 3090",
    "recommended_prefetch_window": 2,
    "recommended_vram_budget_gb": 20,
    "notes": "Higher bandwidth allows tighter prefetch window"
}
```

---

## Which Platform to Choose

Both validate the concept. The choice is driven by budget and timeline, not architecture.

| If you want... | Choose |
|----------------|--------|
| Cheapest path to running benchmarks | AM4/DDR4 |
| Best long-term platform for distillation + future work | AM5/DDR5 |
| Results that reflect future hardware trends | AM5/DDR5 |
| Results that are most representative of average user hardware | AM4/DDR4 |

**Recommendation:** Build on whatever is available first. Run the full benchmark suite on both when possible. The performance delta between platforms is meaningful data, not a problem.

---

## Notes on the RTX 3090 Specifically

The RTX 3090 is a PCIe 4.0 card. Placing it in a PCIe 5.0 slot (AM5) gives it PCIe 4.0 speeds — the card itself is the ceiling, not the slot. This means:

- AM4 PCIe 4.0 and AM5 PCIe 5.0 will show nearly identical GPU transfer speeds with the 3090
- The DDR5 RAM bandwidth improvement is real but the PCIe benefit of AM5 is not realized with this GPU
- This changes somewhat if a PCIe 5.0 GPU is added later

**Practical conclusion:** For Holster Memory benchmarks specifically, AM4/DDR4 and AM5/DDR5 with a 3090 will produce similar PCIe numbers. The DDR5 bandwidth improvement on the RAM side is the meaningful variable.
