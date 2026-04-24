"""
Transcribe audio files with the full pipeline.

Usage:
    # Single file → stdout
    python scripts/transcribe.py audio/speech.wav

    # Save as SRT
    python scripts/transcribe.py audio/speech.wav --output out.srt --format srt

    # Multi-speaker noisy audio, large model
    python scripts/transcribe.py lecture.mp3 \
        --model large-v3 --noise-reduction --format json --output lecture.json

    # Batch directory
    python scripts/transcribe.py --input-dir audio/ --output-dir transcripts/ --format txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.pipeline import TranscriptionPipeline, PipelineConfig


FORMATS = ('txt', 'srt', 'vtt', 'json', 'tsv')


def parse_args():
    p = argparse.ArgumentParser(description='Whisper-based transcription pipeline')
    p.add_argument('input',         nargs='?', default=None,
                   help='Audio file to transcribe')
    p.add_argument('--input-dir',   default=None,
                   help='Directory of audio files for batch processing')
    p.add_argument('--output',      default=None,
                   help='Output file (default: print to stdout)')
    p.add_argument('--output-dir',  default='transcripts',
                   help='Output directory for batch mode')
    p.add_argument('--format',      choices=FORMATS, default='txt')
    p.add_argument('--model',       default='small',
                   choices=('tiny','base','small','medium','large','large-v2','large-v3'))
    p.add_argument('--language',    default=None,
                   help='Force language code (e.g. en, fr, de). Auto-detect if omitted.')
    p.add_argument('--device',      default=None)
    p.add_argument('--chunk-strategy', choices=('fixed','vad'), default='fixed')
    p.add_argument('--chunk-duration', type=float, default=25.0)
    p.add_argument('--noise-reduction', action='store_true')
    p.add_argument('--no-word-timestamps', action='store_true')
    p.add_argument('--beam-size',   type=int,   default=5)
    p.add_argument('--temperature', type=float, default=0.0)
    return p.parse_args()


def build_pipeline(args) -> TranscriptionPipeline:
    cfg = PipelineConfig(
        model_size       = args.model,
        device           = args.device,
        chunk_strategy   = args.chunk_strategy,
        chunk_duration   = args.chunk_duration,
        noise_reduction  = args.noise_reduction,
        language         = args.language,
        word_timestamps  = not args.no_word_timestamps,
        beam_size        = args.beam_size,
        temperature      = args.temperature,
    )
    return TranscriptionPipeline(cfg)


def transcribe_file(pipeline, path, fmt, output_path=None):
    result = pipeline.process_file(path)
    if fmt == 'srt':
        text = result.to_srt()
    elif fmt == 'vtt':
        text = result.to_vtt()
    elif fmt == 'json':
        text = result.to_json()
    else:
        text = result.full_text

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved → {output_path}  "
              f"[lang={result.language} | RTF={result.rtf:.3f} | "
              f"chunks={result.n_chunks}]")
    else:
        print(text)

    return result


def main():
    args     = parse_args()
    pipeline = build_pipeline(args)

    if args.input_dir:
        exts  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4', '.webm'}
        files = [
            os.path.join(args.input_dir, f)
            for f in sorted(os.listdir(args.input_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not files:
            print(f"No audio files found in {args.input_dir}")
            return
        os.makedirs(args.output_dir, exist_ok=True)
        for fpath in files:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            out  = os.path.join(args.output_dir, f"{stem}.{args.format}")
            transcribe_file(pipeline, fpath, args.format, out)

    elif args.input:
        transcribe_file(pipeline, args.input, args.format, args.output)

    else:
        print("Provide an input file or --input-dir.")


if __name__ == '__main__':
    main()
