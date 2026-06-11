"""
metrics.py — HolsterMetrics

Measures the things that matter:
- Did the workload run when it otherwise wouldn't?
- How much of the time was the GPU computing vs waiting?
- How many stalls occurred?
- What was the VRAM hit rate?
"""

import time
import json
import torch
from pathlib import Path
from datetime import datetime


class HolsterMetrics:
    """
    Instruments Holster Memory runtime behavior.

    Key measurements:
        compute_time    — time GPU spent on actual forward passes
        transfer_time   — time spent moving layers to/from VRAM
        stall_count     — times a layer wasn't ready when needed
        vram_hits       — times a layer was already in VRAM (no transfer)
        vram_misses     — times a layer had to be loaded from RAM
        peak_vram_bytes — maximum VRAM allocated during run
        total_tokens    — tokens generated
        generation_time — wall clock time for full generation
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.compute_time = 0.0
        self.transfer_time = 0.0
        self.stall_count = 0
        self.vram_hits = 0
        self.vram_misses = 0
        self.peak_vram_bytes = 0
        self.total_tokens = 0
        self._gen_start = None
        self._gen_end = None
        self._compute_t0 = None
        self._transfer_t0 = None

    # ── Generation lifecycle ──────────────────────────────────────────────

    def generation_start(self):
        self._gen_start = time.perf_counter()

    def generation_end(self):
        self._gen_end = time.perf_counter()
        self.peak_vram_bytes = torch.cuda.max_memory_allocated()

    @property
    def generation_time(self) -> float:
        if self._gen_start and self._gen_end:
            return self._gen_end - self._gen_start
        return 0.0

    # ── Compute timing ────────────────────────────────────────────────────

    def compute_start(self):
        torch.cuda.synchronize()
        self._compute_t0 = time.perf_counter()

    def compute_end(self):
        torch.cuda.synchronize()
        if self._compute_t0:
            self.compute_time += time.perf_counter() - self._compute_t0
            self._compute_t0 = None

    # ── Transfer timing ───────────────────────────────────────────────────

    def record_transfer_start(self):
        self._transfer_t0 = time.perf_counter()
        self.vram_misses += 1

    def record_transfer_end(self):
        if self._transfer_t0:
            self.transfer_time += time.perf_counter() - self._transfer_t0
            self._transfer_t0 = None

    # ── Cache accounting ──────────────────────────────────────────────────

    def record_vram_hit(self):
        self.vram_hits += 1

    def record_stall(self):
        """A layer wasn't in VRAM when the GPU needed it."""
        self.stall_count += 1

    # ── Derived metrics ───────────────────────────────────────────────────

    @property
    def total_accesses(self) -> int:
        return self.vram_hits + self.vram_misses

    @property
    def vram_hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.vram_hits / self.total_accesses

    @property
    def overlap_efficiency(self) -> float:
        """
        Fraction of time spent computing (vs transferring).
        1.0 = perfect overlap (transfers never stalled compute)
        0.0 = fully serial (all transfer time was blocking)
        """
        total = self.compute_time + self.transfer_time
        if total == 0:
            return 0.0
        return self.compute_time / total

    @property
    def tokens_per_second(self) -> float:
        if self.generation_time == 0 or self.total_tokens == 0:
            return 0.0
        return self.total_tokens / self.generation_time

    @property
    def peak_vram_gb(self) -> float:
        return self.peak_vram_bytes / (1024 ** 3)

    # ── Reporting ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "generation_time_s": round(self.generation_time, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_tokens": self.total_tokens,
            "compute_time_s": round(self.compute_time, 3),
            "transfer_time_s": round(self.transfer_time, 3),
            "overlap_efficiency_pct": round(self.overlap_efficiency * 100, 1),
            "stall_count": self.stall_count,
            "vram_hit_rate_pct": round(self.vram_hit_rate * 100, 1),
            "vram_hits": self.vram_hits,
            "vram_misses": self.vram_misses,
            "peak_vram_gb": round(self.peak_vram_gb, 2),
        }

    def report(self, profile: dict = None):
        s = self.summary()
        print("\n" + "=" * 60)
        print("  HOLSTER MEMORY — BENCHMARK RESULTS")
        print("=" * 60)
        if profile:
            print(f"  Platform:          {profile.get('platform', 'unknown')} / {profile.get('ram_type', 'unknown')}")
            print(f"  GPU:               {profile.get('gpu', 'unknown')}")
            print(f"  VRAM Budget:       {profile.get('recommended_vram_budget_gb', '?')} GB")
        print(f"  Generation time:   {s['generation_time_s']}s")
        print(f"  Tokens/sec:        {s['tokens_per_second']}")
        print(f"  Total tokens:      {s['total_tokens']}")
        print("-" * 60)
        print(f"  Compute time:      {s['compute_time_s']}s")
        print(f"  Transfer time:     {s['transfer_time_s']}s")
        print(f"  Overlap efficiency:{s['overlap_efficiency_pct']}%  (higher = better)")
        print("-" * 60)
        print(f"  VRAM hit rate:     {s['vram_hit_rate_pct']}%  (higher = better)")
        print(f"  Stalls:            {s['stall_count']}  (lower = better)")
        print(f"  Peak VRAM used:    {s['peak_vram_gb']} GB")
        print("=" * 60 + "\n")
        return s

    def save(self, path: str, profile: dict = None):
        """Save metrics + profile to a JSON results file."""
        out = {"metrics": self.summary()}
        if profile:
            out["hardware_profile"] = profile
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {path}")
