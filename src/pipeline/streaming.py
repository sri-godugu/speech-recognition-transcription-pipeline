"""Real-time streaming transcription pipeline with ring buffer."""
from __future__ import annotations
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import List, Callable, Optional

import numpy as np

from ..audio.preprocessor import normalize_audio
from ..audio.chunker import StreamingChunker, AudioChunk
from ..asr.whisper_model import WhisperModel
from ..asr.transcriber import Transcriber, Segment
from ..asr.alignment import deduplicate_segments

WHISPER_SR = 16_000


@dataclass
class StreamEvent:
    event_type: str        # 'partial' | 'final' | 'error'
    text:       str        = ''
    segments:   List[Segment] = field(default_factory=list)
    timestamp:  float      = field(default_factory=time.time)
    latency_ms: float      = 0.0


class StreamingTranscriber:
    """
    Simulates real-time microphone transcription over an audio file.

    Architecture:
        Audio source  →  StreamingChunker  →  inference queue
                                                      ↓
                                           Whisper (background thread)
                                                      ↓
                                            StreamEvent callbacks
    """

    def __init__(self, model: WhisperModel,
                 chunk_duration: float = 10.0,
                 language: str = None,
                 on_partial: Callable[[StreamEvent], None] = None,
                 on_final:   Callable[[StreamEvent], None] = None):
        self.transcriber = Transcriber(model, language=language,
                                       word_timestamps=True, temperature=0.0)
        self.chunker     = StreamingChunker(chunk_duration, WHISPER_SR)
        self.on_partial  = on_partial or (lambda e: None)
        self.on_final    = on_final   or (lambda e: None)
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._history: List[Segment] = []

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)

    def feed(self, audio_frame: np.ndarray):
        """Push a frame of audio (must be 16 kHz mono float32)."""
        ready_chunks = self.chunker.feed(audio_frame)
        for chunk in ready_chunks:
            self._queue.put(chunk)

    def flush(self):
        """Process remaining buffered audio."""
        chunk = self.chunker.flush()
        if chunk is not None:
            self._queue.put(chunk)
        self._queue.join()

    def simulate_from_file(self, audio: np.ndarray,
                            frame_duration_s: float = 0.5,
                            realtime: bool = False):
        """
        Feed an audio array as if it were arriving from a microphone.

        If realtime=True, sleeps between frames to simulate real time.
        """
        frame_len = int(frame_duration_s * WHISPER_SR)
        self.start()
        for i in range(0, len(audio), frame_len):
            frame = audio[i: i + frame_len]
            self.feed(frame)
            if realtime:
                time.sleep(frame_duration_s)
        self.flush()
        self.stop()

    # ── Internal ─────────────────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:
                self._queue.task_done()
                break

            t0 = time.perf_counter()
            try:
                audio = normalize_audio(chunk.audio)
                segs  = self.transcriber.transcribe_chunk(
                    audio, time_offset=chunk.start_s, chunk_id=chunk.chunk_id
                )
                latency = (time.perf_counter() - t0) * 1000
                self._history.extend(segs)
                self._history = deduplicate_segments(self._history)

                text = ' '.join(s.text for s in segs)
                full = ' '.join(s.text for s in self._history)

                if segs:
                    self.on_partial(StreamEvent('partial', text, segs,
                                                latency_ms=latency))
                    self.on_final(StreamEvent('final', full, self._history,
                                              latency_ms=latency))
            except Exception as exc:
                self.on_final(StreamEvent('error', str(exc)))
            finally:
                self._queue.task_done()

    @property
    def full_transcript(self) -> str:
        return ' '.join(s.text for s in self._history).strip()
