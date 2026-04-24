"""Audio chunking strategies for long-form transcription."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Iterator

from .vad import EnergyVAD, SpeechSegment

WHISPER_MAX_DURATION = 30.0   # Whisper's hard limit per chunk


@dataclass
class AudioChunk:
    audio:    np.ndarray
    start_s:  float
    end_s:    float
    chunk_id: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class FixedSizeChunker:
    """
    Splits audio into overlapping fixed-duration windows.

    Overlap lets the transcriber cover words that fall on chunk boundaries.
    """

    def __init__(self, chunk_duration: float = 25.0, overlap: float = 2.0,
                 sample_rate: int = 16000):
        self.chunk_dur = min(chunk_duration, WHISPER_MAX_DURATION)
        self.overlap   = overlap
        self.sr        = sample_rate

    def chunk(self, audio: np.ndarray) -> List[AudioChunk]:
        chunks   = []
        step     = self.chunk_dur - self.overlap
        total_s  = len(audio) / self.sr
        start    = 0.0
        chunk_id = 0
        while start < total_s:
            end     = min(start + self.chunk_dur, total_s)
            s_samp  = int(start * self.sr)
            e_samp  = int(end   * self.sr)
            chunks.append(AudioChunk(audio[s_samp:e_samp], start, end, chunk_id))
            chunk_id += 1
            if end >= total_s:
                break
            start += step
        return chunks


class VADChunker:
    """
    Groups VAD speech segments into chunks no longer than max_duration.

    Produces cleaner chunk boundaries by only cutting at silence.
    """

    def __init__(self, vad: EnergyVAD = None,
                 max_duration: float = 25.0,
                 pad_s: float = 0.1,
                 sample_rate: int = 16000):
        self.vad       = vad or EnergyVAD(sample_rate)
        self.max_dur   = min(max_duration, WHISPER_MAX_DURATION)
        self.pad       = pad_s
        self.sr        = sample_rate

    def chunk(self, audio: np.ndarray) -> List[AudioChunk]:
        segments = self.vad.apply(audio)
        if not segments:
            return []

        chunks:   List[AudioChunk] = []
        chunk_id = 0
        group_start = segments[0].start_s
        group_end   = segments[0].end_s

        def flush(start_s, end_s):
            nonlocal chunk_id
            s = max(0, int((start_s - self.pad) * self.sr))
            e = min(len(audio), int((end_s   + self.pad) * self.sr))
            chunks.append(AudioChunk(audio[s:e], start_s, end_s, chunk_id))
            chunk_id += 1

        for seg in segments[1:]:
            if (seg.end_s - group_start) <= self.max_dur:
                group_end = seg.end_s
            else:
                flush(group_start, group_end)
                group_start = seg.start_s
                group_end   = seg.end_s

        flush(group_start, group_end)
        return chunks


class StreamingChunker:
    """
    Yields chunks from a streaming audio source via a ring buffer.

    Call feed() with new audio frames; consume() yields ready chunks.
    """

    def __init__(self, chunk_duration: float = 10.0,
                 sample_rate: int = 16000):
        self.chunk_len = int(chunk_duration * sample_rate)
        self.sr        = sample_rate
        self._buffer   = np.array([], dtype=np.float32)
        self._offset_s = 0.0
        self._chunk_id = 0

    def feed(self, audio: np.ndarray) -> List[AudioChunk]:
        self._buffer = np.concatenate([self._buffer, audio])
        ready = []
        while len(self._buffer) >= self.chunk_len:
            chunk_audio       = self._buffer[:self.chunk_len]
            end_s             = self._offset_s + self.chunk_len / self.sr
            ready.append(AudioChunk(chunk_audio, self._offset_s, end_s, self._chunk_id))
            self._offset_s   += self.chunk_len / self.sr
            self._buffer      = self._buffer[self.chunk_len:]
            self._chunk_id   += 1
        return ready

    def flush(self) -> AudioChunk | None:
        if len(self._buffer) == 0:
            return None
        end_s = self._offset_s + len(self._buffer) / self.sr
        chunk = AudioChunk(self._buffer.copy(), self._offset_s, end_s, self._chunk_id)
        self._buffer   = np.array([], dtype=np.float32)
        self._offset_s = end_s
        self._chunk_id += 1
        return chunk
