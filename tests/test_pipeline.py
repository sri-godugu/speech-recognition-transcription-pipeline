"""
Integration tests for the transcription pipeline (mock Whisper).
Run with: pytest tests/test_pipeline.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.asr.transcriber import Segment, Word, TranscriptionResult
from src.asr.alignment import merge_chunk_segments, deduplicate_segments
from src.pipeline.pipeline import TranscriptionPipeline, PipelineConfig, PipelineResult
from src.postprocessing.formatter import TranscriptionFormatter


SR = 16_000


def _make_audio(duration_s=3.0):
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _mock_transcribe_chunk(audio, time_offset=0.0, chunk_id=0):
    return [Segment(
        text     = 'the quick brown fox',
        start_s  = time_offset,
        end_s    = time_offset + len(audio) / SR,
        words    = [Word('the', time_offset, time_offset + 0.2),
                    Word('quick', time_offset + 0.2, time_offset + 0.5)],
        chunk_id = chunk_id,
    )]


# ── PipelineResult ───────────────────────────────────────────────────────

class TestPipelineResult:
    def _make_result(self):
        segs = [
            Segment('Hello world.', 0.0, 1.5),
            Segment('How are you?', 1.8, 3.2),
        ]
        return PipelineResult(
            segments       = segs,
            full_text      = 'Hello world. How are you?',
            language       = 'en',
            audio_duration = 10.0,
            inference_time = 2.0,
            n_chunks       = 1,
        )

    def test_rtf_computed(self):
        r = self._make_result()
        assert r.rtf == pytest.approx(0.2)

    def test_to_srt_valid(self):
        srt = self._make_result().to_srt()
        assert '00:00:00,000' in srt
        assert '-->' in srt

    def test_to_vtt_valid(self):
        vtt = self._make_result().to_vtt()
        assert vtt.startswith('WEBVTT')

    def test_to_json_valid(self):
        import json
        data = json.loads(self._make_result().to_json())
        assert data['language'] == 'en'
        assert len(data['segments']) == 2

    def test_save_txt(self, tmp_path):
        r    = self._make_result()
        path = str(tmp_path / 'out.txt')
        r.save(path, fmt='txt')
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert 'Hello world.' in content

    def test_save_srt(self, tmp_path):
        r    = self._make_result()
        path = str(tmp_path / 'out.srt')
        r.save(path, fmt='srt')
        assert os.path.exists(path)


# ── TranscriptionPipeline (mocked Whisper) ──────────────────────────────

class TestTranscriptionPipelineMocked:
    def _build_pipeline(self):
        cfg = PipelineConfig(model_size='small', chunk_duration=2.0,
                             overlap=0.2, noise_reduction=False)
        pipeline = TranscriptionPipeline(cfg)
        pipeline._transcriber.transcribe_chunk = _mock_transcribe_chunk
        pipeline._detect_language = lambda audio: 'en'
        return pipeline

    def test_process_array_returns_result(self):
        pipeline = self._build_pipeline()
        audio    = _make_audio(5.0)
        result   = pipeline.process_array(audio)
        assert isinstance(result, PipelineResult)
        assert result.language == 'en'
        assert result.audio_duration == pytest.approx(5.0, abs=0.1)
        assert len(result.segments) > 0

    def test_full_text_non_empty(self):
        pipeline = self._build_pipeline()
        audio    = _make_audio(3.0)
        result   = pipeline.process_array(audio)
        assert result.full_text != ''

    def test_n_chunks_positive(self):
        pipeline = self._build_pipeline()
        audio    = _make_audio(6.0)
        result   = pipeline.process_array(audio)
        assert result.n_chunks >= 1

    def test_segments_chronological(self):
        pipeline = self._build_pipeline()
        audio    = _make_audio(8.0)
        result   = pipeline.process_array(audio)
        starts   = [s.start_s for s in result.segments]
        assert starts == sorted(starts)

    def test_noise_reduction_flag(self):
        cfg = PipelineConfig(model_size='small', noise_reduction=True,
                             chunk_duration=2.0, overlap=0.2)
        pipeline = TranscriptionPipeline(cfg)
        pipeline._transcriber.transcribe_chunk = _mock_transcribe_chunk
        pipeline._detect_language = lambda audio: 'en'
        audio  = _make_audio(3.0)
        result = pipeline.process_array(audio)
        assert isinstance(result, PipelineResult)

    def test_vad_chunk_strategy(self):
        cfg = PipelineConfig(model_size='small', chunk_strategy='vad',
                             chunk_duration=2.0)
        pipeline = TranscriptionPipeline(cfg)
        pipeline._transcriber.transcribe_chunk = _mock_transcribe_chunk
        pipeline._detect_language = lambda audio: 'en'
        audio  = _make_audio(3.0)
        result = pipeline.process_array(audio)
        assert isinstance(result, PipelineResult)


# ── Formatter round-trip ─────────────────────────────────────────────────

class TestFormatterRoundTrip:
    def test_srt_vtt_both_have_timestamps(self):
        segs = [Segment('hello', 0.0, 1.0), Segment('world', 1.5, 2.5)]
        srt  = TranscriptionFormatter.to_srt(segs)
        vtt  = TranscriptionFormatter.to_vtt(segs)
        assert '00:00:01,000' in srt
        assert '00:00:01.000' in vtt

    def test_json_preserves_all_segments(self):
        import json
        segs = [Segment(f'word{i}', float(i), float(i+1)) for i in range(5)]
        data = json.loads(TranscriptionFormatter.to_json(segs))
        assert len(data['segments']) == 5
