"""
Tests for audio processing utilities.
Run with: pytest tests/test_audio.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from src.audio.preprocessor import (
    normalize_audio, rms_normalize, pre_emphasis,
    spectral_subtraction, add_noise,
)
from src.audio.vad import EnergyVAD, SpeechSegment, apply_vad_filter
from src.audio.chunker import FixedSizeChunker, VADChunker, StreamingChunker


SR = 16000


def _sine(freq=440, dur=1.0, sr=SR) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(dur=1.0, sr=SR) -> np.ndarray:
    return np.zeros(int(dur * sr), dtype=np.float32)


# ── Preprocessor ────────────────────────────────────────────────────────

class TestNormalizeAudio:
    def test_peak_at_target(self):
        audio = _sine() * 0.1
        norm  = normalize_audio(audio, target_peak=0.95)
        assert np.abs(norm).max() == pytest.approx(0.95, abs=1e-4)

    def test_silent_unchanged(self):
        audio = np.zeros(1000, dtype=np.float32)
        norm  = normalize_audio(audio)
        assert np.allclose(norm, 0)

    def test_returns_float32(self):
        audio = _sine()
        assert normalize_audio(audio).dtype == np.float32


class TestPreEmphasis:
    def test_output_length(self):
        audio = _sine()
        out   = pre_emphasis(audio)
        assert len(out) == len(audio)

    def test_reduces_low_freq(self):
        # Pre-emphasis should reduce energy at the first sample relative to original
        low_freq = np.ones(100, dtype=np.float32) * 0.5
        out = pre_emphasis(low_freq, coeff=0.97)
        assert out[1] < low_freq[1]


class TestAddNoise:
    def test_output_shape(self):
        audio = _sine()
        noisy = add_noise(audio, snr_db=20)
        assert noisy.shape == audio.shape

    def test_values_clipped(self):
        audio = _sine()
        noisy = add_noise(audio, snr_db=-10)    # very noisy
        assert noisy.max() <= 1.0
        assert noisy.min() >= -1.0

    def test_snr_affects_noise_level(self):
        audio       = _sine()
        low_snr     = add_noise(audio, snr_db=5)
        high_snr    = add_noise(audio, snr_db=30)
        diff_low    = np.mean((low_snr  - audio) ** 2)
        diff_high   = np.mean((high_snr - audio) ** 2)
        assert diff_low > diff_high


class TestSpectralSubtraction:
    def test_output_shape(self):
        audio = _sine()
        out   = spectral_subtraction(audio, SR)
        assert len(out) == len(audio)

    def test_reduces_noise(self):
        clean = _sine(440, 2.0)
        noisy = add_noise(clean, snr_db=5)
        denoised = spectral_subtraction(noisy, SR)
        # Denoised should be closer to clean than noisy
        dist_noisy    = np.mean((noisy    - clean) ** 2)
        dist_denoised = np.mean((denoised - clean) ** 2)
        assert dist_denoised < dist_noisy


# ── VAD ─────────────────────────────────────────────────────────────────

class TestEnergyVAD:
    def setup_method(self):
        self.vad = EnergyVAD(SR, energy_threshold=0.01, min_speech_ms=100)

    def test_detects_speech(self):
        speech = _sine(440, 2.0)
        segs   = self.vad.apply(speech)
        assert len(segs) >= 1

    def test_silence_returns_no_segments(self):
        silence = _silence(2.0)
        segs    = self.vad.apply(silence)
        assert len(segs) == 0

    def test_segment_timestamps_valid(self):
        audio = _sine(440, 3.0)
        segs  = self.vad.apply(audio)
        for seg in segs:
            assert seg.start_s >= 0
            assert seg.end_s   <= 3.0 + 0.1  # small tolerance for padding
            assert seg.start_s <  seg.end_s

    def test_speech_then_silence(self):
        speech  = _sine(440, 1.0)
        silence = _silence(1.0)
        audio   = np.concatenate([speech, silence])
        segs    = self.vad.apply(audio)
        assert len(segs) >= 1
        assert segs[0].start_s < 0.5

    def test_mask_shape(self):
        audio = _sine(440, 1.0)
        mask  = self.vad.get_speech_mask(audio)
        assert mask.shape == audio.shape
        assert mask.dtype == bool


class TestApplyVADFilter:
    def test_filters_silence(self):
        speech  = _sine(440, 1.0)
        silence = _silence(1.0)
        audio   = np.concatenate([speech, silence])
        vad     = EnergyVAD(SR, energy_threshold=0.01, min_speech_ms=100)
        segs    = vad.apply(audio)
        filtered = apply_vad_filter(audio, segs, SR)
        assert len(filtered) < len(audio)

    def test_empty_segments_returns_empty(self):
        audio    = _sine()
        filtered = apply_vad_filter(audio, [], SR)
        assert len(filtered) == 0


# ── Chunker ──────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_n_chunks(self):
        audio   = _sine(440, 100.0)   # 100 seconds
        chunker = FixedSizeChunker(chunk_duration=25.0, overlap=2.0)
        chunks  = chunker.chunk(audio)
        assert len(chunks) >= 4

    def test_chunk_ids_sequential(self):
        audio  = _sine(440, 60.0)
        chunks = FixedSizeChunker(25.0, 2.0).chunk(audio)
        ids    = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_timestamps_cover_audio(self):
        audio  = _sine(440, 30.0)
        chunks = FixedSizeChunker(25.0, 2.0).chunk(audio)
        assert chunks[0].start_s == pytest.approx(0.0)
        assert chunks[-1].end_s  == pytest.approx(30.0, abs=0.1)

    def test_chunk_does_not_exceed_max(self):
        audio  = _sine(440, 120.0)
        chunks = FixedSizeChunker(25.0, 2.0).chunk(audio)
        for c in chunks:
            assert c.duration_s <= 25.1   # tiny tolerance

    def test_short_audio_single_chunk(self):
        audio  = _sine(440, 5.0)
        chunks = FixedSizeChunker(25.0, 2.0).chunk(audio)
        assert len(chunks) == 1


class TestStreamingChunker:
    def test_emit_on_fill(self):
        chunker = StreamingChunker(chunk_duration=1.0, sample_rate=SR)
        frame   = _sine(440, 0.5)
        c1 = chunker.feed(frame)
        assert len(c1) == 0
        c2 = chunker.feed(frame)
        assert len(c2) == 1

    def test_flush_returns_remainder(self):
        chunker = StreamingChunker(chunk_duration=1.0, sample_rate=SR)
        frame   = _sine(440, 0.3)
        chunker.feed(frame)
        rem = chunker.flush()
        assert rem is not None
        assert rem.duration_s == pytest.approx(0.3, abs=0.01)

    def test_empty_flush(self):
        chunker = StreamingChunker(chunk_duration=1.0, sample_rate=SR)
        assert chunker.flush() is None
