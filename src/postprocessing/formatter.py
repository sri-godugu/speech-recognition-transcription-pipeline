"""Output formatters: SRT, WebVTT, JSON, plain text."""
from __future__ import annotations
import json
from typing import List

from ..asr.transcriber import Segment


class TranscriptionFormatter:

    @staticmethod
    def to_srt(segments: List[Segment]) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{_srt_time(seg.start_s)} --> {_srt_time(seg.end_s)}")
            lines.append(seg.text)
            lines.append('')
        return '\n'.join(lines)

    @staticmethod
    def to_vtt(segments: List[Segment]) -> str:
        lines = ['WEBVTT', '']
        for seg in segments:
            lines.append(f"{_vtt_time(seg.start_s)} --> {_vtt_time(seg.end_s)}")
            lines.append(seg.text)
            lines.append('')
        return '\n'.join(lines)

    @staticmethod
    def to_json(segments: List[Segment],
                language: str = 'en',
                full_text: str = '') -> str:
        data = {
            'language': language,
            'text':     full_text,
            'segments': [
                {
                    'id':      i,
                    'start':   round(seg.start_s, 3),
                    'end':     round(seg.end_s,   3),
                    'text':    seg.text,
                    'words':   [
                        {
                            'word':        w.text,
                            'start':       round(w.start_s, 3),
                            'end':         round(w.end_s,   3),
                            'probability': round(w.probability, 4),
                        }
                        for w in seg.words
                    ],
                }
                for i, seg in enumerate(segments)
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def to_txt(segments: List[Segment]) -> str:
        return '\n'.join(seg.text for seg in segments)

    @staticmethod
    def to_tsv(segments: List[Segment]) -> str:
        rows = ['start\tend\ttext']
        for seg in segments:
            rows.append(f"{seg.start_s:.3f}\t{seg.end_s:.3f}\t{seg.text}")
        return '\n'.join(rows)


# ── Helpers ─────────────────────────────────────────────────────────────

def _srt_time(seconds: float) -> str:
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
