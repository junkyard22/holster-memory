"""
holster.py — Core HolsterModel implementation

Treats VRAM as an active execution cache. Streams model layers from
system RAM (the Holster) into VRAM on demand, prefetching ahead of
compute and evicting cold layers to stay within a hard VRAM budget.
"""

import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from scheduler import LayerScheduler
from metrics import HolsterMetrics


def load_profile(profile_path: str) -> dict:
    with open(profile_path) as f:
        return json.load(f)


class HolsterModel:
    """
    A VRAM-budget-aware inference wrapper that streams transformer layers
    from system RAM into VRAM on demand.

    Args:
        model_name:       HuggingFace model identifier or local path
        vram_budget_gb:   Hard VRAM ceiling in gigabytes
        prefetch_window:  Number of layers to prefetch ahead of current
        profile_path:     Optional path to hardware profile JSON
    """

    def __init__(
        self,
        model_name: str,
        vram_budget_gb: float = 20.0,
        prefetch_window: int = 3,
        profile_path: str = None,
    ):
        self.model_name = model_name
        self.vram_budget_bytes = int(vram_budget_gb * 1024 ** 3)
        self.device = torch.device("cuda")
        self.metrics = HolsterMetrics()

        # Load hardware profile if provided
        self.profile = load_profile(profile_path) if profile_path else {}
        if self.profile:
            prefetch_window = self.profile.get("recommended_prefetch_window", prefetch_window)
            vram_budget_gb = self.profile.get("recommended_vram_budget_gb", vram_budget_gb)
            self.vram_budget_bytes = int(vram_budget_gb * 1024 ** 3)

        print(f"Loading {model_name} to CPU (Holster)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Identify transformer blocks
        self.blocks = self._find_transformer_blocks()
        self.num_layers = len(self.blocks)
        print(f"Found {self.num_layers} transformer layers.")

        # Scheduler
        self.scheduler = LayerScheduler(
            num_layers=self.num_layers,
            prefetch_window=prefetch_window,
        )

        # Layer tracking
        self.active_layers: dict[int, nn.Module] = {}

        # CUDA streams — one for compute, one for async prefetch
        self.compute_stream = torch.cuda.Stream()
        self.prefetch_stream = torch.cuda.Stream()

        print(
            f"Holster ready. VRAM budget: {vram_budget_gb}GB | "
            f"Prefetch window: {prefetch_window} | "
            f"Platform: {self.profile.get('platform', 'unknown')}"
        )

    def _find_transformer_blocks(self):
        """Locate the transformer block list inside the model."""
        # Covers LLaMA, Mistral, Phi, Qwen, Gemma, Falcon, MPT families
        candidates = [
            lambda m: m.model.layers,
            lambda m: m.transformer.h,
            lambda m: m.gpt_neox.layers,
            lambda m: m.model.decoder.layers,
        ]
        for getter in candidates:
            try:
                blocks = getter(self.model)
                if len(blocks) > 0:
                    return blocks
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate transformer blocks. "
            "Add a custom getter for this model architecture."
        )

    def _vram_used_bytes(self) -> int:
        return torch.cuda.memory_allocated()

    def _layer_fits_in_budget(self) -> bool:
        return self._vram_used_bytes() < self.vram_budget_bytes

    def _load_layer(self, idx: int):
        """Move layer idx from CPU (Holster) to VRAM. No-op if already loaded."""
        if idx in self.active_layers:
            self.metrics.record_vram_hit()
            return

        self.metrics.record_transfer_start()
        with torch.cuda.stream(self.prefetch_stream):
            self.blocks[idx].to(self.device, non_blocking=True)
            self.active_layers[idx] = self.blocks[idx]
        self.metrics.record_transfer_end()

    def _evict_layer(self, idx: int):
        """Move layer idx from VRAM back to CPU (Holster)."""
        if idx not in self.active_layers:
            return
        # Wait for both streams to finish with this layer before evicting
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_stream(self.prefetch_stream)
        self.active_layers[idx].to("cpu")
        del self.active_layers[idx]
        torch.cuda.empty_cache()

    def _prefetch_window(self, current_idx: int):
        """Async-prefetch the next N layers while current layer computes."""
        to_load = self.scheduler.get_next_to_load(current_idx)
        with torch.cuda.stream(self.prefetch_stream):
            for idx in to_load:
                self._load_layer(idx)

    def _evict_cold_layers(self, current_idx: int):
        """Evict layers that have fallen outside the active window."""
        candidates = self.scheduler.get_candidates_to_evict(
            current_idx, list(self.active_layers.keys())
        )
        for idx in candidates:
            self._evict_layer(idx)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate tokens with Holster Memory active."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generated = input_ids.clone()

        self.metrics.reset()
        self.metrics.generation_start()

        for token_step in range(max_new_tokens):
            current_input = generated[:, -1:].to(self.device)

            # Embed
            hidden = self.model.model.embed_tokens(current_input)

            # Layer-by-layer forward pass with streaming
            for i in range(self.num_layers):
                # Prefetch upcoming layers while we compute current
                self._prefetch_window(i)

                # Ensure current layer is loaded
                if i not in self.active_layers:
                    self.metrics.record_stall()
                    self._load_layer(i)

                # Wait for prefetch to finish before compute
                with torch.cuda.stream(self.compute_stream):
                    self.compute_stream.wait_stream(self.prefetch_stream)
                    self.metrics.compute_start()
                    layer_out = self.active_layers[i](hidden)
                    # Some models return tuples, some return tensors
                    hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    self.metrics.compute_end()

                # Evict layers that are now cold
                self._evict_cold_layers(i)

            # Decode
            logits = self.model.lm_head(hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).cpu()
            generated = torch.cat([generated, next_token], dim=-1)

        self.metrics.generation_end()
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def report(self):
        """Print a full metrics report for this run."""
        self.metrics.report(profile=self.profile)
