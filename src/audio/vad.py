"""Voice Activity Detection (energy + ZCR based)."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpeechSegment:
    start_s: float
    end_s:   float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class EnergyVAD:
    """
    Frame-level VAD using short-term energy and zero-crossing rate.

    Decision rule: frame is speech if energy > threshold AND zcr < max_zcr.
    Temporal smoothing removes short spurious speech/silence bursts.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 frame_ms: int = 20,
                 energy_threshold: float = 0.02,
                 zcr_threshold: float = 0.3,
                 speech_pad_ms: int = 200,
                 min_speech_ms: int = 250,
                 min_silence_ms: int = 100):
        self.sr              = sample_rate
        self.frame_len       = int(sample_rate * frame_ms / 1000)
        self.energy_thr      = energy_threshold
        self.zcr_thr         = zcr_threshold
        self.pad_frames      = int(speech_pad_ms  / frame_ms)
        self.min_speech_f    = int(min_speech_ms  / frame_ms)
        self.min_silence_f   = int(min_silence_ms / frame_ms)

    def apply(self, audio: np.ndarray) -> List[SpeechSegment]:
        """Return list of SpeechSegments detected in audio."""
        decisions = self._frame_decisions(audio)
        decisions = self._smooth(decisions)
        return self._decisions_to_segments(decisions, len(audio))

    def get_speech_mask(self, audio: np.ndarray) -> np.ndarray:
        """Return sample-level boolean mask (True = speech)."""
        segments = self.apply(audio)
        mask = np.zeros(len(audio), dtype=bool)
        for seg in segments:
            s = int(seg.start_s * self.sr)
            e = int(seg.end_s   * self.sr)
            mask[s:e] = True
        return mask

    # ── Internal ─────────────────────────────────────────────────────────

    def _frame_decisions(self, audio: np.ndarray) -> np.ndarray:
        n_frames = len(audio) // self.frame_len
        decisions = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            frame  = audio[i * self.frame_len: (i + 1) * self.frame_len]
            energy = np.mean(frame ** 2)
            zcr    = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            decisions[i] = (energy > self.energy_thr) and (zcr < self.zcr_thr)
        return decisions

    def _smooth(self, decisions: np.ndarray) -> np.ndarray:
        d = decisions.copy()
        # Fill short silences within speech
        i = 0
        while i < len(d):
            if d[i]:
                i += 1
                continue
            j = i
            while j < len(d) and not d[j]:
                j += 1
            if (j - i) < self.min_silence_f:
                d[i:j] = True
            i = j + 1
        # Remove short speech bursts
        i = 0
        while i < len(d):
            if not d[i]:
                i += 1
                continue
            j = i
            while j < len(d) and d[j]:
                j += 1
            if (j - i) < self.min_speech_f:
                d[i:j] = False
            i = j + 1
        # Apply padding
        padded = d.copy()
        speech_idx = np.where(d)[0]
        for idx in speech_idx:
            lo = max(0, idx - self.pad_frames)
            hi = min(len(d), idx + self.pad_frames + 1)
            padded[lo:hi] = True
        return padded

    def _decisions_to_segments(self, decisions: np.ndarray,
                                n_samples: int) -> List[SpeechSegment]:
        segments: List[SpeechSegment] = []
        in_speech = False
        start = 0
        frame_dur = self.frame_len / self.sr
        for i, is_speech in enumerate(decisions):
            if is_speech and not in_speech:
                start     = i
                in_speech = True
            elif not is_speech and in_speech:
                segments.append(SpeechSegment(start * frame_dur, i * frame_dur))
                in_speech = False
        if in_speech:
            end_s = min(len(decisions) * frame_dur, n_samples / self.sr)
            segments.append(SpeechSegment(start * frame_dur, end_s))
        return segments


def apply_vad_filter(audio: np.ndarray, segments: List[SpeechSegment],
                     sample_rate: int) -> np.ndarray:
    """Concatenate only speech segments from audio."""
    parts = []
    for seg in segments:
        s = int(seg.start_s * sample_rate)
        e = int(seg.end_s   * sample_rate)
        parts.append(audio[s:e])
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
