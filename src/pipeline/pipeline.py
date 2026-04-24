"""End-to-end transcription pipeline orchestration."""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..audio.loader import load_audio, WHISPER_SR
from ..audio.preprocessor import normalize_audio, spectral_subtraction
from ..audio.vad import EnergyVAD
from ..audio.chunker import FixedSizeChunker, VADChunker, AudioChunk
from ..asr.whisper_model import WhisperModel
from ..asr.transcriber import Transcriber, Segment, TranscriptionResult
from ..asr.alignment import (merge_chunk_segments, deduplicate_segments,
                              assign_global_timestamps)
from ..postprocessing.formatter import TranscriptionFormatter
from ..postprocessing.punctuation import capitalize_sentences


@dataclass
class PipelineConfig:
    model_size:        str   = 'small'
    device:            str   = None
    chunk_strategy:    str   = 'fixed'    # 'fixed' | 'vad'
    chunk_duration:    float = 25.0
    overlap:           float = 2.0
    noise_reduction:   bool  = False
    normalize:         bool  = True
    language:          str   = None
    word_timestamps:   bool  = True
    beam_size:         int   = 5
    temperature:       float = 0.0
    restore_punctuation: bool = True


@dataclass
class PipelineResult:
    segments:       List[Segment]
    full_text:      str
    language:       str
    audio_duration: float
    inference_time: float
    n_chunks:       int

    @property
    def rtf(self) -> float:
        return self.inference_time / max(self.audio_duration, 1e-7)

    def to_srt(self) -> str:
        return TranscriptionFormatter.to_srt(self.segments)

    def to_vtt(self) -> str:
        return TranscriptionFormatter.to_vtt(self.segments)

    def to_json(self) -> str:
        return TranscriptionFormatter.to_json(self.segments, self.language, self.full_text)

    def save(self, path: str, fmt: str = 'txt'):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if fmt == 'srt':
            text = self.to_srt()
        elif fmt == 'vtt':
            text = self.to_vtt()
        elif fmt == 'json':
            text = self.to_json()
        else:
            text = self.full_text
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)


class TranscriptionPipeline:
    """
    Configurable end-to-end speech transcription pipeline.

    Steps:
        1. Load audio (any format → 16 kHz mono float32)
        2. Preprocess (normalize, optional noise reduction)
        3. Chunk (fixed-size or VAD-based)
        4. Transcribe each chunk (Whisper)
        5. Merge & deduplicate segments
        6. Post-process (punctuation restoration)
    """

    def __init__(self, config: PipelineConfig = None):
        self.cfg        = config or PipelineConfig()
        self._model     = WhisperModel(self.cfg.model_size, self.cfg.device)
        self._transcriber = Transcriber(
            self._model,
            language        = self.cfg.language,
            word_timestamps = self.cfg.word_timestamps,
            beam_size       = self.cfg.beam_size,
            temperature     = self.cfg.temperature,
        )

    def process_file(self, audio_path: str) -> PipelineResult:
        audio, duration = self._load_and_preprocess(audio_path)
        return self._run(audio, duration)

    def process_array(self, audio: np.ndarray,
                       sample_rate: int = WHISPER_SR) -> PipelineResult:
        if sample_rate != WHISPER_SR:
            import torchaudio, torch
            wav = torch.from_numpy(audio).unsqueeze(0)
            wav = torchaudio.functional.resample(wav, sample_rate, WHISPER_SR)
            audio = wav.squeeze(0).numpy()
        if self.cfg.normalize:
            audio = normalize_audio(audio)
        duration = len(audio) / WHISPER_SR
        return self._run(audio, duration)

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_and_preprocess(self, path: str):
        from ..audio.loader import get_audio_info
        info     = get_audio_info(path)
        duration = info['duration_s']
        audio    = load_audio(path)
        if self.cfg.normalize:
            audio = normalize_audio(audio)
        if self.cfg.noise_reduction:
            audio = spectral_subtraction(audio, WHISPER_SR)
        return audio, duration

    def _run(self, audio: np.ndarray, duration: float) -> PipelineResult:
        chunks = self._chunk(audio)
        t0     = time.perf_counter()

        all_segs: List[List[Segment]] = []
        for chunk in chunks:
            segs = self._transcriber.transcribe_chunk(
                chunk.audio, time_offset=chunk.start_s, chunk_id=chunk.chunk_id
            )
            all_segs.append(segs)

        elapsed  = time.perf_counter() - t0
        segments = merge_chunk_segments(all_segs, self.cfg.overlap)
        segments = deduplicate_segments(segments)
        segments = assign_global_timestamps(segments)

        if self.cfg.restore_punctuation:
            for seg in segments:
                seg.text = capitalize_sentences(seg.text)

        full_text = ' '.join(s.text for s in segments).strip()
        language  = self._detect_language(audio)

        return PipelineResult(
            segments       = segments,
            full_text      = full_text,
            language       = language,
            audio_duration = duration,
            inference_time = elapsed,
            n_chunks       = len(chunks),
        )

    def _chunk(self, audio: np.ndarray) -> List[AudioChunk]:
        if self.cfg.chunk_strategy == 'vad':
            vad     = EnergyVAD(WHISPER_SR)
            chunker = VADChunker(vad, max_duration=self.cfg.chunk_duration)
        else:
            chunker = FixedSizeChunker(self.cfg.chunk_duration,
                                        self.cfg.overlap, WHISPER_SR)
        chunks = chunker.chunk(audio)
        return chunks if chunks else [
            __import__('src.audio.chunker', fromlist=['AudioChunk']).AudioChunk(
                audio, 0.0, len(audio) / WHISPER_SR, 0
            )
        ]

    def _detect_language(self, audio: np.ndarray) -> str:
        if self.cfg.language:
            return self.cfg.language
        try:
            lang, _ = self._model.detect_language(audio[:WHISPER_SR * 30])
            return lang
        except Exception:
            return 'en'
