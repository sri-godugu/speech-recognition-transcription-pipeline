"""Core transcription logic: chunk → Whisper → structured segments."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .whisper_model import WhisperModel


@dataclass
class Word:
    text:       str
    start_s:    float
    end_s:      float
    probability: float = 1.0


@dataclass
class Segment:
    text:     str
    start_s:  float
    end_s:    float
    words:    List[Word] = field(default_factory=list)
    chunk_id: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class TranscriptionResult:
    segments:       List[Segment]
    language:       str
    full_text:      str
    audio_duration: float
    inference_time: float

    @property
    def rtf(self) -> float:
        return self.inference_time / max(self.audio_duration, 1e-7)


class Transcriber:
    """
    Transcribes individual audio chunks using a WhisperModel.

    Returns structured Segment objects with per-word timestamps.
    """

    def __init__(self, model: WhisperModel,
                 language: str = None,
                 word_timestamps: bool = True,
                 beam_size: int = 5,
                 temperature: float = 0.0):
        self.model           = model
        self.language        = language
        self.word_timestamps = word_timestamps
        self.beam_size       = beam_size
        self.temperature     = temperature

    def transcribe_chunk(self, audio: np.ndarray,
                         time_offset: float = 0.0,
                         chunk_id: int = 0) -> List[Segment]:
        """
        Transcribe one audio chunk.

        time_offset shifts all timestamps to global audio time.
        """
        if len(audio) == 0:
            return []

        result = self.model.transcribe(
            audio,
            language=self.language,
            word_timestamps=self.word_timestamps,
            beam_size=self.beam_size,
            temperature=self.temperature,
        )

        segments = []
        for raw_seg in result.get('segments', []):
            words = []
            for w in raw_seg.get('words', []):
                words.append(Word(
                    text        = w['word'],
                    start_s     = w['start'] + time_offset,
                    end_s       = w['end']   + time_offset,
                    probability = w.get('probability', 1.0),
                ))
            segments.append(Segment(
                text     = raw_seg['text'].strip(),
                start_s  = raw_seg['start'] + time_offset,
                end_s    = raw_seg['end']   + time_offset,
                words    = words,
                chunk_id = chunk_id,
            ))
        return segments

    def transcribe_audio(self, audio: np.ndarray,
                         sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe a complete (short) audio array in one pass."""
        duration = len(audio) / sample_rate
        t0       = time.perf_counter()
        result   = self.model.transcribe(
            audio,
            language=self.language,
            word_timestamps=self.word_timestamps,
            beam_size=self.beam_size,
            temperature=self.temperature,
        )
        elapsed = time.perf_counter() - t0

        segments = self.transcribe_chunk(audio)
        return TranscriptionResult(
            segments       = segments,
            language       = result.get('language', 'en'),
            full_text      = result.get('text', '').strip(),
            audio_duration = duration,
            inference_time = elapsed,
        )
