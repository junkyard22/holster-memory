"""
benchmark.py — Holster Memory CLI Benchmark Runner

Runs a generation benchmark and compares:
  1. Baseline: HuggingFace device_map="auto" (standard CPU offload)
  2. Holster:  HolsterModel (tiered VRAM cache with scheduler)

Usage:
    python benchmark.py --profile ../profiles/am4_ddr4.json --model 3b
    python benchmark.py --profile ../profiles/am5_ddr5.json --model 7b --tokens 100
    python benchmark.py --model-name meta-llama/Llama-3.2-3B-Instruct --vram 16
"""

import argparse
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add harness dir to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from holster import HolsterModel


# ── Model size shortcuts ───────────────────────────────────────────────────

MODEL_SHORTCUTS = {
    "1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "7b":  "meta-llama/Llama-3.1-8B-Instruct",
    "8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "13b": "meta-llama/Llama-2-13b-chat-hf",
}

DEFAULT_PROMPT = (
    "Explain the concept of memory caching in computer architecture. "
    "Why does cache hierarchy matter for performance?"
)


# ── Baseline runner ───────────────────────────────────────────────────────

def run_baseline(model_name: str, prompt: str, max_new_tokens: int) -> dict:
    """
    Baseline: load model with device_map='auto' (HuggingFace default offload).
    This is what users get without Holster Memory.
    """
    print("\nRunning BASELINE (device_map='auto')...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]

    result = {
        "mode": "baseline_device_map_auto",
        "generation_time_s": round(elapsed, 3),
        "tokens_generated": tokens_generated,
        "tokens_per_second": round(tokens_generated / elapsed, 2) if elapsed > 0 else 0,
        "peak_vram_gb": round(peak_vram, 2),
        "oom": False,
    }
    print(f"  Baseline: {result['tokens_per_second']} tok/s | "
          f"Peak VRAM: {result['peak_vram_gb']}GB | "
          f"Time: {result['generation_time_s']}s")
    return result


# ── Holster runner ────────────────────────────────────────────────────────

def run_holster(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    vram_budget_gb: float,
    prefetch_window: int,
    profile: dict,
    profile_path: str,
) -> dict:
    """
    Holster Memory: tiered VRAM cache with scheduler.
    """
    print(f"\nRunning HOLSTER (budget={vram_budget_gb}GB, window={prefetch_window})...")
    torch.cuda.reset_peak_memory_stats()

    model = HolsterModel(
        model_name=model_name,
        vram_budget_gb=vram_budget_gb,
        prefetch_window=prefetch_window,
        profile_path=profile_path,
    )

    output = model.generate(prompt, max_new_tokens=max_new_tokens)
    metrics = model.metrics.summary()
    metrics["mode"] = "holster_memory"
    metrics["oom"] = False

    model.report()
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Holster Memory Benchmark")
    parser.add_argument("--model", default="3b", help="Model shortcut (1b/3b/7b/8b/13b) or HF model name")
    parser.add_argument("--model-name", default=None, help="Full HuggingFace model name (overrides --model)")
    parser.add_argument("--profile", default=None, help="Path to hardware profile JSON")
    parser.add_argument("--vram", type=float, default=20.0, help="VRAM budget in GB")
    parser.add_argument("--window", type=int, default=3, help="Prefetch window size")
    parser.add_argument("--tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Generation prompt")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline run")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    # Resolve model name
    model_name = args.model_name or MODEL_SHORTCUTS.get(args.model, args.model)
    print(f"\nHolster Memory Benchmark")
    print(f"Model:  {model_name}")
    print(f"Tokens: {args.tokens}")
    print(f"Prompt: {args.prompt[:80]}...")

    # Load profile
    profile = {}
    if args.profile:
        with open(args.profile) as f:
            profile = json.load(f)
        print(f"Profile: {profile.get('profile_name', args.profile)}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt": args.prompt,
        "max_new_tokens": args.tokens,
        "hardware_profile": profile,
        "baseline": None,
        "holster": None,
    }

    # Baseline
    if not args.skip_baseline:
        try:
            results["baseline"] = run_baseline(model_name, args.prompt, args.tokens)
        except torch.cuda.OutOfMemoryError:
            print("  Baseline OOM — model too large for VRAM without offload")
            results["baseline"] = {"mode": "baseline_device_map_auto", "oom": True}

    # Holster
    try:
        results["holster"] = run_holster(
            model_name=model_name,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            vram_budget_gb=args.vram,
            prefetch_window=args.window,
            profile=profile,
            profile_path=args.profile,
        )
    except torch.cuda.OutOfMemoryError:
        print("  Holster OOM — check VRAM budget setting")
        results["holster"] = {"mode": "holster_memory", "oom": True}

    # Comparison summary
    if results["baseline"] and results["holster"]:
        b = results["baseline"]
        h = results["holster"]
        if not b.get("oom") and not h.get("oom"):
            ratio = h.get("tokens_per_second", 0) / max(b.get("tokens_per_second", 1), 0.001)
            print(f"\n{'='*60}")
            print(f"  COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"  Baseline tok/s:  {b.get('tokens_per_second', 'N/A')}")
            print(f"  Holster tok/s:   {h.get('tokens_per_second', 'N/A')}")
            print(f"  Speed ratio:     {ratio:.2f}x")
            print(f"  Baseline VRAM:   {b.get('peak_vram_gb', 'N/A')} GB")
            print(f"  Holster VRAM:    {h.get('peak_vram_gb', 'N/A')} GB")
            print(f"  Holster stalls:  {h.get('stall_count', 'N/A')}")
            print(f"  Overlap eff:     {h.get('overlap_efficiency_pct', 'N/A')}%")
            print(f"{'='*60}\n")

    # Save results
    output_path = args.output
    if not output_path:
        platform_tag = profile.get("platform", "unknown").lower()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"benchmarks/results/{platform_tag}_{ts}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
