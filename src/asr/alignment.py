"""Post-processing: merge overlapping segments, dedup, assign word timestamps."""
from __future__ import annotations
from typing import List

from .transcriber import Segment, Word


def merge_chunk_segments(chunks_segments: List[List[Segment]],
                          overlap_s: float = 2.0) -> List[Segment]:
    """
    Merge per-chunk segment lists into a single chronological list.

    Removes duplicate text that appears in the overlap region between
    consecutive chunks (Whisper repeats words near chunk boundaries).
    """
    if not chunks_segments:
        return []

    merged: List[Segment] = list(chunks_segments[0])

    for chunk_segs in chunks_segments[1:]:
        if not chunk_segs:
            continue
        cutoff = merged[-1].end_s - overlap_s if merged else 0.0
        for seg in chunk_segs:
            if seg.start_s >= cutoff - 0.05:
                merged.append(seg)

    return merged


def deduplicate_segments(segments: List[Segment],
                          similarity_threshold: float = 0.8) -> List[Segment]:
    """Remove near-identical consecutive segments caused by chunking overlap."""
    if not segments:
        return []
    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if _text_overlap(prev.text, seg.text) < similarity_threshold:
            result.append(seg)
    return result


def assign_global_timestamps(segments: List[Segment]) -> List[Segment]:
    """Ensure segments are sorted by start time."""
    return sorted(segments, key=lambda s: s.start_s)


def segments_to_words(segments: List[Segment]) -> List[Word]:
    """Flatten all word-level timestamps from a segment list."""
    words = []
    for seg in segments:
        words.extend(seg.words)
    return sorted(words, key=lambda w: w.start_s)


def fill_missing_word_timestamps(segment: Segment) -> Segment:
    """
    Linearly distribute timestamps for words that lack them.
    Happens when Whisper returns a segment without per-word info.
    """
    if segment.words:
        return segment
    tokens  = segment.text.split()
    dur     = segment.duration_s / max(len(tokens), 1)
    words   = []
    for i, tok in enumerate(tokens):
        words.append(Word(
            text     = tok,
            start_s  = segment.start_s + i * dur,
            end_s    = segment.start_s + (i + 1) * dur,
        ))
    segment.words = words
    return segment


# ── Internal helpers ────────────────────────────────────────────────────

def _text_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word sets between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
