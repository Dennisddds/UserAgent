"""RTWI-style adaptive percentile thresholds: replace fixed hyperparameters with
per-user rolling-window percentile calibration.

Motivation (from RTWI paper): fixed global thresholds don't adapt to user difficulty.
Hard users (low reliability overall) need lower thresholds to avoid filtering everything;
easy users (high reliability) can use higher thresholds for precise anomaly detection.

Usage:
    tracker = AdaptiveThresholds(window_size=50, alpha=0.3)
    ...
    # Record observed values
    tracker.record("coverage", 0.65)
    tracker.record("confidence", 0.72)
    ...
    # Get adaptive thresholds
    min_coverage = tracker.percentile("coverage")  # 30th percentile of recent coverage
    min_confidence = tracker.percentile("confidence")

If not enough history (< 10 samples), falls back to configured defaults.
"""

from __future__ import annotations

from collections import deque
from typing import Any


# Default percentile α: filter the bottom α fraction
DEFAULT_ALPHA = 0.30  # 30th percentile = filter bottom 30%

# Fields we track with their fallback defaults
TRACKED_FIELDS: dict[str, float] = {
    "coverage": 0.35,           # evidence coverage per step
    "theory_confidence": 0.35,  # theory routing min_confidence
    "mining_score": 0.35,       # stage reliability mining score
    "synthesis_score": 0.30,    # stage reliability synthesis score
    "surprise": 0.60,           # surprise/fast-path threshold
    "repair_overlap": 0.28,     # failure structure overlap for repair matching
}


class AdaptiveThresholds:
    """Per-user rolling-window percentile tracker.

    Each tracked field maintains a deque of recent values. When percentile()
    is called with fewer than `min_samples` data points, the hardcoded
    fallback is returned instead — small-sample users don't get unreliable
    adaptive thresholds.
    """

    def __init__(
        self,
        window_size: int = 50,
        alpha: float = DEFAULT_ALPHA,
        min_samples: int = 10,
        fallbacks: dict[str, float] | None = None,
    ) -> None:
        self.window_size = window_size
        self.alpha = alpha
        self.min_samples = min_samples
        self.fallbacks = dict(fallbacks or TRACKED_FIELDS)
        self._buffers: dict[str, deque[float]] = {}

    def record(self, field: str, value: float) -> None:
        if field not in self._buffers:
            self._buffers[field] = deque(maxlen=self.window_size)
        self._buffers[field].append(value)

    def percentile(self, field: str, alpha: float | None = None) -> float:
        """Return the α-percentile of recent values (lower α = more strict).

        For 'surprise', we return the (1-α) percentile because high surprise
        → slow path, so we want the UPPER tail.
        """
        a = alpha if alpha is not None else self.alpha
        buf = self._buffers.get(field)
        if not buf or len(buf) < self.min_samples:
            return self.fallbacks.get(field, 0.5)
        invert = field in {"surprise"}
        k = int(len(buf) * (1.0 - a)) if invert else int(len(buf) * a)
        k = max(0, min(len(buf) - 1, k))
        sorted_vals = sorted(buf)
        return round(sorted_vals[k], 4)

    def record_from_trace(self, c_trace: dict[str, Any]) -> None:
        """Bulk-record from c_trace stage_reliability block."""
        sr = c_trace.get("stage_reliability") or {}
        mining = sr.get("mining") or {}
        synth = sr.get("synthesis") or {}
        coverage = mining.get("coverage")
        mining_score = mining.get("score")
        synth_score = synth.get("score")
        if coverage is not None:
            self.record("coverage", float(coverage))
        if mining_score is not None:
            self.record("mining_score", float(mining_score))
        if synth_score is not None:
            self.record("synthesis_score", float(synth_score))

    def record_surprise(self, surprise: float) -> None:
        self.record("surprise", float(surprise))

    def record_theory_confidence(self, confidences: list[float]) -> None:
        for c in confidences:
            self.record("theory_confidence", float(c))

    def record_repair_overlap(self, overlaps: list[float]) -> None:
        for ov in overlaps:
            self.record("repair_overlap", float(ov))

    def summary(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "window_size": self.window_size,
            "fields": {
                f: {
                    "n": len(buf),
                    "adaptive": len(buf) >= self.min_samples,
                    "percentile": self.percentile(f),
                    "fallback": self.fallbacks.get(f, 0.5),
                }
                for f, buf in self._buffers.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        return {
            "alpha": self.alpha,
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "buffers": {f: list(buf) for f, buf in self._buffers.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AdaptiveThresholds":
        at = cls(
            window_size=int(d.get("window_size", 50)),
            alpha=float(d.get("alpha", DEFAULT_ALPHA)),
            min_samples=int(d.get("min_samples", 10)),
        )
        for f, vals in (d.get("buffers") or {}).items():
            at._buffers[f] = deque(vals, maxlen=at.window_size)
        return at
