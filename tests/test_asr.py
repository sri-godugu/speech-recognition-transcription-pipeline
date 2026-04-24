"""
Tests for ASR components: alignment, formatter, punctuation, metrics.
Run with: pytest tests/test_asr.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from src.asr.transcriber import Segment, Word
from src.asr.alignment import (
    merge_chunk_segments, deduplicate_segments,
    assign_global_timestamps, segments_to_words,
    fill_missing_word_timestamps, _text_overlap,
)
from src.postprocessing.formatter import (
    TranscriptionFormatter, _srt_time, _vtt_time,
)
from src.postprocessing.punctuation import (
    capitalize_sentences, ensure_terminal_punctuation,
    remove_filler_words, normalize_whitespace,
)
from src.utils.metrics import (
    word_error_rate, character_error_rate,
    compute_rtf, wer_breakdown,
)


def make_seg(text, start, end, words=None):
    return Segment(text=text, start_s=start, end_s=end,
                   words=words or [], chunk_id=0)


# ── Alignment ────────────────────────────────────────────────────────────

class TestMergeChunkSegments:
    def test_single_chunk(self):
        segs   = [[make_seg('hello world', 0, 1)]]
        merged = merge_chunk_segments(segs)
        assert len(merged) == 1
        assert merged[0].text == 'hello world'

    def test_two_non_overlapping_chunks(self):
        c1 = [make_seg('hello', 0, 1)]
        c2 = [make_seg('world', 1.5, 2.5)]
        merged = merge_chunk_segments([c1, c2])
        assert len(merged) == 2

    def test_empty_input(self):
        assert merge_chunk_segments([]) == []

    def test_empty_chunks_skipped(self):
        c1 = [make_seg('hello', 0, 1)]
        c2 = []
        merged = merge_chunk_segments([c1, c2])
        assert len(merged) == 1


class TestDeduplicateSegments:
    def test_identical_segments_removed(self):
        segs = [
            make_seg('the cat sat', 0, 1),
            make_seg('the cat sat', 0.9, 1.9),
        ]
        result = deduplicate_segments(segs, similarity_threshold=0.8)
        assert len(result) == 1

    def test_different_segments_kept(self):
        segs = [
            make_seg('the cat sat', 0, 1),
            make_seg('on the mat', 1.1, 2.1),
        ]
        result = deduplicate_segments(segs)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate_segments([]) == []


class TestTextOverlap:
    def test_identical(self):
        assert _text_overlap('hello world', 'hello world') == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _text_overlap('hello', 'world') == pytest.approx(0.0)

    def test_partial(self):
        overlap = _text_overlap('the cat sat', 'the cat ran')
        assert 0 < overlap < 1


class TestFillMissingWordTimestamps:
    def test_distributes_evenly(self):
        seg   = make_seg('a b c d', 0.0, 2.0)
        seg   = fill_missing_word_timestamps(seg)
        assert len(seg.words) == 4
        assert seg.words[0].start_s == pytest.approx(0.0)
        assert seg.words[-1].end_s  == pytest.approx(2.0, abs=0.01)

    def test_already_has_words_unchanged(self):
        words = [Word('hello', 0.0, 0.5)]
        seg   = make_seg('hello', 0.0, 0.5, words=words)
        seg   = fill_missing_word_timestamps(seg)
        assert len(seg.words) == 1


# ── Formatter ────────────────────────────────────────────────────────────

class TestSRTTime:
    def test_zero(self):
        assert _srt_time(0.0) == '00:00:00,000'

    def test_one_hour(self):
        assert _srt_time(3600.0) == '01:00:00,000'

    def test_fractional(self):
        assert _srt_time(1.5) == '00:00:01,500'


class TestVTTTime:
    def test_format(self):
        assert _vtt_time(0.0) == '00:00:00.000'

    def test_one_minute(self):
        assert _vtt_time(60.0) == '00:01:00.000'


class TestTranscriptionFormatter:
    def setup_method(self):
        self.segs = [
            make_seg('Hello world.', 0.0, 1.5),
            make_seg('How are you?', 2.0, 3.5),
        ]

    def test_srt_has_index(self):
        srt = TranscriptionFormatter.to_srt(self.segs)
        assert '1\n' in srt
        assert '2\n' in srt

    def test_srt_has_timestamps(self):
        srt = TranscriptionFormatter.to_srt(self.segs)
        assert '-->' in srt

    def test_vtt_starts_with_webvtt(self):
        vtt = TranscriptionFormatter.to_vtt(self.segs)
        assert vtt.startswith('WEBVTT')

    def test_json_is_valid(self):
        import json
        data = json.loads(TranscriptionFormatter.to_json(
            self.segs, 'en', 'Hello world. How are you?'))
        assert 'segments' in data
        assert len(data['segments']) == 2

    def test_txt_plain(self):
        txt = TranscriptionFormatter.to_txt(self.segs)
        assert 'Hello world.' in txt

    def test_tsv_header(self):
        tsv = TranscriptionFormatter.to_tsv(self.segs)
        assert tsv.startswith('start\tend\ttext')


# ── Punctuation ──────────────────────────────────────────────────────────

class TestCapitalizeSentences:
    def test_first_letter_capitalized(self):
        assert capitalize_sentences('hello world')[0] == 'H'

    def test_after_period(self):
        result = capitalize_sentences('hello. world')
        assert 'W' in result

    def test_empty(self):
        assert capitalize_sentences('') == ''


class TestFillerRemoval:
    def test_removes_um(self):
        result = remove_filler_words('um, I think uh it is')
        assert 'um' not in result.lower()
        assert 'uh' not in result.lower()

    def test_preserves_words(self):
        result = remove_filler_words('I think it is fine')
        assert 'think' in result
        assert 'fine' in result


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace('hello   world') == 'hello world'

    def test_strips_edges(self):
        assert normalize_whitespace('  hi  ') == 'hi'


# ── Metrics ──────────────────────────────────────────────────────────────

class TestWER:
    def test_perfect(self):
        assert word_error_rate('hello world', 'hello world') == pytest.approx(0.0)

    def test_all_wrong(self):
        wer = word_error_rate('abc def', 'xyz uvw')
        assert wer == pytest.approx(1.0)

    def test_one_substitution(self):
        wer = word_error_rate('hello there', 'hello world')
        assert 0 < wer <= 0.5 + 1e-5

    def test_empty_hypothesis(self):
        wer = word_error_rate('', 'hello world')
        assert wer == pytest.approx(1.0)

    def test_empty_reference(self):
        wer = word_error_rate('hello', '')
        assert wer == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert word_error_rate('Hello World', 'hello world') == pytest.approx(0.0)

    def test_extra_word_insertion(self):
        wer = word_error_rate('the big cat sat', 'the cat sat')
        assert wer > 0


class TestCER:
    def test_perfect(self):
        assert character_error_rate('hello', 'hello') == pytest.approx(0.0)

    def test_finer_than_wer(self):
        wer = word_error_rate('hell', 'hello')
        cer = character_error_rate('hell', 'hello')
        assert cer < wer


class TestWERBreakdown:
    def test_substitution_counted(self):
        bd = wer_breakdown('cat sat on mat', 'cat sat on hat')
        assert bd['substitutions'] == 1
        assert bd['deletions']     == 0
        assert bd['insertions']    == 0

    def test_deletion_counted(self):
        bd = wer_breakdown('cat sat', 'cat sat on mat')
        assert bd['deletions'] == 2


class TestRTF:
    def test_faster_than_realtime(self):
        assert compute_rtf(10.0, 2.0) == pytest.approx(0.2)

    def test_zero_audio(self):
        rtf = compute_rtf(0.0, 1.0)
        assert rtf > 0
