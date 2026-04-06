# Holster Memory — Architecture

## Overview

Holster Memory is a tiered GPU memory architecture for consumer AI inference. It treats VRAM as a high-speed execution cache, not a fixed container, and manages a larger passive RAM pool as the primary workload store.

The system has four components:

| Component | Responsibility |
|-----------|---------------|
| **Scheduler** | Predicts upcoming layer access, decides eviction candidates |
| **Transfer Manager** | Executes async PCIe copies via CUDA streams |
| **VRAM Pool Manager** | Enforces hard VRAM budget, tracks allocations |
| **Metrics Collector** | Measures compute/transfer overlap, stall rate, hit rate |

## Memory Tiers

```
┌─────────────────────────────────┐
│         Active VRAM             │  ← executing now
│  current layer + hot window     │    fastest, smallest
│  KV cache (pinned)              │
└────────────────┬────────────────┘
                 │ PCIe (async, non-blocking)
┌────────────────▼────────────────┐
│       Passive RAM (Holster)     │  ← staged, warm
│  full model weights             │    slower, much larger
│  upcoming layers queued         │
└────────────────┬────────────────┘
                 │ disk I/O (last resort)
┌────────────────▼────────────────┐
│         Cold Storage            │  ← reload only
│  model files on disk            │    slowest, unlimited
└─────────────────────────────────┘
```

## Execution Flow

For each token generation step:

1. **Embed** — input token embedded on GPU (always in VRAM)
2. **For each layer i:**
   a. Prefetch stream: async load layers [i+1 … i+window] from RAM → VRAM
   b. Compute stream: wait for prefetch, run forward pass on layer i
   c. Evict layers outside [i-tail … i+window] back to RAM
3. **Decode** — LM head produces logits (always in VRAM)
4. **Sample** — next token selected, appended to sequence

The key invariant: compute stream never starts a layer until the prefetch stream has confirmed it's loaded. This is enforced via `compute_stream.wait_stream(prefetch_stream)`.

## Scheduler Design

The scheduler maintains a **hot window** around the current execution position:

```
hot window = [current - tail, current + prefetch_window]
```

Default values:
- `prefetch_window = 3` (load 3 layers ahead)
- `tail = 1` (keep 1 already-computed layer, handles skip connections)

Maximum simultaneous VRAM residents = `tail + 1 + prefetch_window` = 5 layers.

Layer size varies by model. For a 7B model with 32 layers in fp16:
- ~14GB total / 32 layers ≈ 437MB per layer
- 5 active layers ≈ 2.2GB of weight VRAM usage
- Remainder of VRAM budget available for KV cache, embeddings, LM head

## KV Cache Handling

KV cache is **not streamed**. It lives in VRAM and is managed by the model's native attention implementation. This is intentional — the KV cache is accessed randomly (any past token position), making sequential streaming inappropriate.

VRAM budget accounting must reserve space for the KV cache:
```
available_for_weights = vram_budget - kv_cache_reserved - embeddings_lm_head
```

At long sequence lengths, KV cache growth can exceed layer weight VRAM. If this becomes a constraint, options include:
- Sliding window attention (discard old tokens)
- KV cache quantization (fp8 or int4 cache)
- CPU KV cache paging (vLLM-style, future work)

## CUDA Stream Architecture

Two dedicated streams:

| Stream | Purpose |
|--------|---------|
| `prefetch_stream` | Async PCIe transfers (CPU RAM → VRAM) |
| `compute_stream` | GPU forward pass execution |

Synchronization rule:
```python
compute_stream.wait_stream(prefetch_stream)  # before every layer forward
```

Eviction waits on both streams:
```python
compute_stream.wait_stream(prefetch_stream)  # before moving layer back to CPU
```

This prevents the race condition where eviction begins before compute has finished using the layer.

## Metrics That Matter

Success is measured in two tiers:

**Tier 1 (correctness):** Does the workload run when it otherwise would not?
- OOM rate = 0%
- VRAM never exceeds budget

**Tier 2 (efficiency):** Does it run with acceptable slowdown?
- Overlap efficiency > 50% (compute > transfer time)
- Stall count < 5 per generation
- Tokens/sec competitive with baseline CPU offload

## Known Limitations

**PCIe bandwidth ceiling:** The 3090 is PCIe 4.0, giving ~28 GB/s practical transfer speed. A 7B model layer (~437MB) takes ~16ms to transfer. At low compute speeds (slow models, long sequences), this is the dominant cost.

**HuggingFace forward loop:** HF's `generate()` assumes all layers on the same device. HolsterModel rewrites the forward loop manually. This means it does not automatically benefit from HF optimizations like Flash Attention unless those are explicitly wired in.

**KV cache growth:** Long contexts can exhaust the VRAM budget regardless of weight streaming. Reserve adequate KV cache headroom in the VRAM budget config.

**First-token latency:** The first token requires loading layer 0 from scratch. Subsequent tokens benefit from the warm Holster (layers are evicted back to RAM, not to disk, so reload is fast).
