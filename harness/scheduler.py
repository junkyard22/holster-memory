"""
scheduler.py — Layer Scheduler for Holster Memory

Decides which layers to prefetch and which to evict based on
the current execution position and a configurable lookahead window.
"""


class LayerScheduler:
    """
    Manages promotion and eviction decisions for transformer layers.

    The scheduler's job is to keep the GPU fed without blowing the
    VRAM budget. It does this by maintaining a sliding hot window
    around the current execution position.

    Hot window = [current - tail, current + prefetch_window]

    Layers inside the window are kept in VRAM.
    Layers outside the window are eviction candidates.

    Args:
        num_layers:       Total number of transformer layers in the model
        prefetch_window:  How many layers ahead to prefetch
        tail:             How many already-computed layers to retain
                          (useful if model has skip connections)
    """

    def __init__(self, num_layers: int, prefetch_window: int = 3, tail: int = 1):
        self.num_layers = num_layers
        self.prefetch_window = prefetch_window
        self.tail = tail

    def get_next_to_load(self, current_idx: int) -> list[int]:
        """
        Returns layer indices to prefetch given current execution position.
        Only returns indices that exist (no out-of-bounds).
        """
        start = current_idx + 1
        end = min(current_idx + self.prefetch_window + 1, self.num_layers)
        return list(range(start, end))

    def get_candidates_to_evict(
        self, current_idx: int, active_layers: list[int]
    ) -> list[int]:
        """
        Returns layer indices that are loaded but outside the hot window.
        These are safe to evict.

        Hot window = [current - tail, current + prefetch_window]
        """
        keep_start = max(0, current_idx - self.tail)
        keep_end = min(self.num_layers, current_idx + self.prefetch_window + 1)
        hot_window = set(range(keep_start, keep_end))
        return [idx for idx in active_layers if idx not in hot_window]

    def hot_window_size(self) -> int:
        """Maximum number of layers that can be in VRAM at once."""
        return self.tail + 1 + self.prefetch_window

    def describe(self) -> str:
        return (
            f"LayerScheduler("
            f"num_layers={self.num_layers}, "
            f"prefetch_window={self.prefetch_window}, "
            f"tail={self.tail}, "
            f"max_active={self.hot_window_size()})"
        )
