"""
Evaluate transcription accuracy against ground truth.

Usage:
    python scripts/evaluate.py \
        --test-list data/test.tsv \
        --model small \
        --output-dir eval_out/

    # Test list format (tab-separated):
    #   /path/to/audio.wav\tground truth transcript

    # With noise augmentation
    python scripts/evaluate.py --test-list data/test.tsv --snr 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.pipeline import TranscriptionPipeline, PipelineConfig
from src.audio.loader import load_audio
from src.audio.preprocessor import add_noise
from src.utils.metrics import word_error_rate, character_error_rate, compute_rtf
from src.utils.visualization import plot_spectrogram


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-list',  required=True,
                   help='TSV: audio_path<tab>reference_text')
    p.add_argument('--model',      default='small')
    p.add_argument('--output-dir', default='eval_out')
    p.add_argument('--snr',        type=float, default=None,
                   help='Add Gaussian noise at this SNR (dB) before transcribing')
    p.add_argument('--noise-reduction', action='store_true')
    p.add_argument('--language',   default=None)
    p.add_argument('--device',     default=None)
    p.add_argument('--max-samples',type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.test_list) as f:
        samples = [l.strip().split('\t') for l in f if '\t' in l]

    if args.max_samples:
        samples = samples[:args.max_samples]

    cfg = PipelineConfig(
        model_size      = args.model,
        device          = args.device,
        language        = args.language,
        noise_reduction = args.noise_reduction,
    )
    pipeline = TranscriptionPipeline(cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    wers, cers, rtfs = [], [], []
    details = []

    for i, (wav_path, ref_text) in enumerate(samples):
        audio = load_audio(wav_path)

        if args.snr is not None:
            audio = add_noise(audio, args.snr)

        result  = pipeline.process_array(audio)
        hyp     = result.full_text
        wer_val = word_error_rate(hyp, ref_text)
        cer_val = character_error_rate(hyp, ref_text)

        wers.append(wer_val)
        cers.append(cer_val)
        rtfs.append(result.rtf)

        details.append({
            'file':      wav_path,
            'reference': ref_text,
            'hypothesis':hyp,
            'wer':       round(wer_val, 4),
            'cer':       round(cer_val, 4),
            'rtf':       round(result.rtf, 4),
        })

        print(f"\r  [{i+1}/{len(samples)}] WER={wer_val*100:.1f}%", end='', flush=True)

    print()

    summary = {
        'model':      args.model,
        'snr_db':     args.snr,
        'n_samples':  len(samples),
        'wer_mean':   float(np.mean(wers)),
        'wer_std':    float(np.std(wers)),
        'cer_mean':   float(np.mean(cers)),
        'rtf_mean':   float(np.mean(rtfs)),
    }

    print(f"\n{'='*55}")
    print(f"  Model   : {args.model}")
    print(f"  Samples : {len(samples)}")
    print(f"  SNR     : {args.snr} dB" if args.snr else "  SNR     : clean")
    print(f"  WER     : {summary['wer_mean']*100:.2f}% ± {summary['wer_std']*100:.2f}%")
    print(f"  CER     : {summary['cer_mean']*100:.2f}%")
    print(f"  RTF     : {summary['rtf_mean']:.4f}")
    print(f"{'='*55}")

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, 'details.json'), 'w') as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
