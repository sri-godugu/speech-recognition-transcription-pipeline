"""
Simulate real-time streaming transcription from an audio file.

Feeds audio in small frames (mimicking a microphone) and prints
partial + final transcripts as they arrive.

Usage:
    python scripts/stream.py audio/speech.wav
    python scripts/stream.py audio/speech.wav --model small --chunk-duration 8 --realtime
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio.loader import load_audio
from src.audio.preprocessor import normalize_audio
from src.asr.whisper_model import WhisperModel
from src.pipeline.streaming import StreamingTranscriber, StreamEvent


CYAN  = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'
RED   = '\033[91m'


def on_partial(event: StreamEvent):
    if event.text.strip():
        print(f"{CYAN}[PARTIAL  {event.timestamp:.1f}s  lat={event.latency_ms:.0f}ms]{RESET} "
              f"{event.text}", flush=True)


def on_final(event: StreamEvent):
    if event.event_type == 'error':
        print(f"{RED}[ERROR]{RESET} {event.text}")
        return
    if event.text.strip():
        print(f"{GREEN}[TRANSCRIPT]{RESET} {event.text}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description='Streaming ASR simulation')
    p.add_argument('input',             help='Audio file to stream')
    p.add_argument('--model',           default='small')
    p.add_argument('--device',          default=None)
    p.add_argument('--language',        default=None)
    p.add_argument('--chunk-duration',  type=float, default=10.0,
                   help='Seconds of audio per inference chunk')
    p.add_argument('--frame-duration',  type=float, default=0.5,
                   help='Microphone frame size in seconds')
    p.add_argument('--realtime',        action='store_true',
                   help='Sleep between frames to simulate real-time input')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading audio: {args.input}")
    audio = normalize_audio(load_audio(args.input))
    duration = len(audio) / 16000
    print(f"Duration: {duration:.1f}s  |  Model: {args.model}  |  "
          f"Chunk: {args.chunk_duration}s  |  Realtime: {args.realtime}")
    print("-" * 60)

    model  = WhisperModel(args.model, args.device)
    model.load()

    streamer = StreamingTranscriber(
        model,
        chunk_duration = args.chunk_duration,
        language       = args.language,
        on_partial     = on_partial,
        on_final       = on_final,
    )

    t_start = time.perf_counter()
    streamer.simulate_from_file(
        audio,
        frame_duration_s = args.frame_duration,
        realtime         = args.realtime,
    )
    elapsed = time.perf_counter() - t_start

    print("-" * 60)
    print(f"Stream complete in {elapsed:.1f}s  |  RTF={elapsed/duration:.3f}")
    print(f"\nFull transcript:\n{streamer.full_transcript}")


if __name__ == '__main__':
    main()
