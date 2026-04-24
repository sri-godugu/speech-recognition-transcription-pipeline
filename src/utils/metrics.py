"""ASR evaluation metrics: WER, CER, RTF, latency tracking."""
from __future__ import annotations
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ── WER / CER ───────────────────────────────────────────────────────────

def word_error_rate(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words.
    Lower is better; 0.0 = perfect.
    """
    h = _normalize(hypothesis).split()
    r = _normalize(reference).split()
    return _levenshtein(h, r) / max(len(r), 1)


def character_error_rate(hypothesis: str, reference: str) -> float:
    """Character Error Rate (same formula as WER but over characters)."""
    h = list(_normalize(hypothesis))
    r = list(_normalize(reference))
    return _levenshtein(h, r) / max(len(r), 1)


def wer_breakdown(hypothesis: str, reference: str) -> dict:
    """Return S/D/I counts along with WER."""
    h = _normalize(hypothesis).split()
    r = _normalize(reference).split()
    dist, ops = _levenshtein_ops(h, r)
    subs = sum(1 for o in ops if o == 'sub')
    dels = sum(1 for o in ops if o == 'del')
    ins  = sum(1 for o in ops if o == 'ins')
    n    = max(len(r), 1)
    return {
        'wer':           dist / n,
        'substitutions': subs,
        'deletions':     dels,
        'insertions':    ins,
        'ref_words':     len(r),
    }


# ── Real-Time Factor ─────────────────────────────────────────────────────

def compute_rtf(audio_duration_s: float, inference_time_s: float) -> float:
    """RTF = inference_time / audio_duration. < 1 means faster than real time."""
    return inference_time_s / max(audio_duration_s, 1e-7)


# ── Latency tracker ──────────────────────────────────────────────────────

@dataclass
class LatencyTracker:
    """
    Measures end-to-end latency for streaming or batch transcription.

    Tracks: preprocessing, chunking, inference, postprocessing, total.
    """
    _stages: dict = field(default_factory=dict)
    _starts: dict = field(default_factory=dict)

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def end(self, stage: str):
        if stage in self._starts:
            elapsed = time.perf_counter() - self._starts.pop(stage)
            self._stages[stage] = self._stages.get(stage, 0.0) + elapsed

    def summary(self) -> dict:
        total = sum(self._stages.values())
        return {
            **{k: round(v * 1000, 2) for k, v in self._stages.items()},
            'total_ms': round(total * 1000, 2),
        }


@dataclass
class BenchmarkResult:
    model_size:     str
    dataset:        str
    condition:      str    # 'clean' | 'noisy_SNR10' | etc.
    wer:            float
    cer:            float
    rtf:            float
    latency_ms:     float
    n_samples:      int
    notes:          str = ''

    def __str__(self):
        return (f"[{self.model_size}|{self.condition}] "
                f"WER={self.wer*100:.1f}% CER={self.cer*100:.1f}% "
                f"RTF={self.rtf:.3f} lat={self.latency_ms:.0f}ms")


# ── Internal helpers ─────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _levenshtein(h: list, r: list) -> int:
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if r[i-1] == h[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def _levenshtein_ops(h: list, r: list):
    m, n = len(r), len(h)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    # Backtrack
    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i-1] == h[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append('sub'); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append('del'); i -= 1
        else:
            ops.append('ins'); j -= 1
    return dp[m][n], ops
