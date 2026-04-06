# Holster Memory — Prior Art & Defensive Publication

## Publication Intent

This document establishes a public, timestamped record of the Holster Memory concept as originally conceived by James Yarber. It is published as a defensive publication to prevent any third party from obtaining exclusive rights to this idea or its implementation.

## Original Conception

**Concept name:** Holster Memory  
**Author:** James Yarber  
**Original conception date:** December 2025  
**First documented:** December 2025 (internal document, Holster_Memory.docx)  
**Public repository established:** 2026  

## Core Claim

Holster Memory is a tiered GPU memory architecture in which:

1. VRAM is treated as a high-speed **active execution cache**, not a fixed container
2. A larger, slower memory pool (system RAM, or PCIe/CXL-attached expansion memory) serves as a **passive staging layer**
3. A **scheduler** manages promotion, demotion, and prefetch of data between tiers
4. The system degrades gracefully under memory pressure rather than failing with OOM

The specific framing — *"VRAM as a cache, not a container"* — and the named architecture (Active VRAM / Passive Holster / Cold Storage + Scheduler) constitute the original conceptual contribution.

## Canon Line

> *Slow memory holsters it. VRAM draws the gun and pulls the trigger.*

## What This Is Not Claiming

This publication does not claim to have invented:
- Memory tiering in general
- CPU cache hierarchies
- GPU memory paging
- Texture streaming
- ZeRO-style optimizer offloading
- Unified memory systems
- Any existing ML framework offload mechanism

The novelty is the **consumer-oriented application** of these principles to GPU inference workloads, with VRAM explicitly reframed as an execution cache and the architecture organized around a named, schedulable passive tier.

## Related Prior Work (Not Invented Here)

| Concept | Where It Exists |
|---------|----------------|
| CPU cache hierarchies | Foundational computer architecture |
| Memory paging | Operating systems since the 1960s |
| Texture streaming | Game engines (Unreal, Unity) |
| Unified memory | NVIDIA CUDA unified memory |
| ZeRO offloading | DeepSpeed (Microsoft Research) |
| device_map="auto" | Hugging Face Accelerate |
| vLLM KV cache paging | vLLM project |

Holster Memory differs from all of the above in that it:
- Targets consumer GPU inference specifically
- Treats the passive tier as a named, first-class architectural component
- Organizes the scheduler around predictable AI layer execution patterns
- Frames the entire system around graceful degradation as the primary value

## Revision History

| Date | Event |
|------|-------|
| December 2025 | Concept originated, internal document created |
| 2026 | Public repository established, prior art documented |
| TBD | Phase 1 prototype complete |
| TBD | Benchmark results published |
