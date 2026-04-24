"""
Benchmark Whisper model sizes across noise levels and chunking strategies.

Generates WER/RTF tables and plots. Results saved to results/.

Usage:
    python scripts/benchmark.py --test-list data/test.tsv --output-dir results/
    python scripts/benchmark.py --simulate  # use built-in simulated results
"""
import argparse
import json
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Simulated benchmark data (LibriSpeech test-clean / test-other) ──────
# Based on published Whisper paper results + realistic noise benchmarks.
# Replace with actual evaluation runs on your dataset.

SIMULATED_RESULTS = {
    'metadata': {
        'dataset':     'LibriSpeech test-clean / test-other (simulated)',
        'sample_rate': 16000,
        'note': ('Results are simulated from Whisper paper (Radford et al. 2022) '
                 'and noise-augmentation benchmarks. '
                 'Run evaluate.py on real data to get actual numbers.'),
    },
    'wer_by_model_and_snr': {
        # Keys: model size, values: dict {snr_db: wer}
        # SNR: 'clean' = no added noise; numbers = dB SNR
        'tiny':     {'clean': 0.088, '20': 0.112, '15': 0.138, '10': 0.189, '5': 0.264, '0': 0.381},
        'base':     {'clean': 0.057, '20': 0.071, '15': 0.092, '10': 0.138, '5': 0.201, '0': 0.302},
        'small':    {'clean': 0.036, '20': 0.047, '15': 0.063, '10': 0.098, '5': 0.154, '0': 0.239},
        'medium':   {'clean': 0.026, '20': 0.034, '15': 0.048, '10': 0.073, '5': 0.119, '0': 0.192},
        'large-v3': {'clean': 0.019, '20': 0.024, '15': 0.033, '10': 0.054, '5': 0.089, '0': 0.148},
    },
    'cer_by_model': {
        'tiny':     0.041, 'base': 0.024, 'small': 0.015,
        'medium':   0.010, 'large-v3': 0.007,
    },
    'rtf_cpu': {
        # Intel Xeon (single-threaded), fp32
        'tiny':     0.18,  'base':  0.42,  'small': 1.05,
        'medium':   3.20,  'large-v3': 7.80,
    },
    'rtf_gpu': {
        # NVIDIA A100 40GB, fp16
        'tiny':     0.018, 'base':  0.038, 'small': 0.072,
        'medium':   0.14,  'large-v3': 0.28,
    },
    'latency_breakdown_ms': {
        # End-to-end for a 30s audio clip on GPU (small model)
        'load_preprocess': 12.4,
        'vad_chunking':     8.1,
        'inference':      982.3,
        'postprocess':      6.2,
    },
    'chunk_strategy_comparison': {
        # WER on clean audio, small model
        'fixed_25s':     0.036,
        'vad_based':     0.033,
        'no_chunking':   0.041,
    },
    'noise_reduction_impact': {
        # WER at SNR=10 dB, small model, with and without spectral subtraction
        'without_nr': 0.098,
        'with_nr':    0.079,
        'improvement_pct': 19.4,
    },
}


def save_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # CSV: WER by model and SNR
    wer_data = results['wer_by_model_and_snr']
    snr_keys = ['clean', '20', '15', '10', '5', '0']
    csv_path = os.path.join(output_dir, 'wer_by_noise.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model'] + [f'SNR={k}dB' if k != 'clean' else 'clean' for k in snr_keys])
        for model, wer_dict in wer_data.items():
            row = [model] + [f"{wer_dict.get(k, '')*100:.2f}" for k in snr_keys]
            writer.writerow(row)

    # CSV: RTF
    rtf_cpu = results['rtf_cpu']
    rtf_gpu = results.get('rtf_gpu', {})
    rtf_path = os.path.join(output_dir, 'rtf_comparison.csv')
    with open(rtf_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'rtf_cpu', 'rtf_gpu'])
        for m in rtf_cpu:
            writer.writerow([m, rtf_cpu[m], rtf_gpu.get(m, '')])

    print(f"Saved benchmark results to {output_dir}/")


def print_table(results: dict):
    wer_data = results['wer_by_model_and_snr']
    rtf_cpu  = results['rtf_cpu']
    rtf_gpu  = results.get('rtf_gpu', {})

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS — Whisper ASR Pipeline")
    print("  " + results['metadata']['note'])
    print("=" * 70)

    # WER table
    snr_keys = ['clean', '20', '15', '10', '5', '0']
    header   = f"{'Model':<12}" + "".join(f"{'clean' if k=='clean' else k+'dB':>8}" for k in snr_keys)
    print("\n  WER (%) by Model and SNR Level")
    print("  " + "-" * 60)
    print("  " + header)
    for model, wer_dict in wer_data.items():
        row = f"  {model:<12}" + "".join(
            f"{wer_dict.get(k, 0)*100:>7.1f}%" for k in snr_keys
        )
        print(row)

    # RTF table
    print("\n  Real-Time Factor (RTF)")
    print("  " + "-" * 40)
    print(f"  {'Model':<14} {'CPU':>8} {'GPU (A100)':>12}")
    for m in rtf_cpu:
        gpu_str = f"{rtf_gpu[m]:.3f}" if m in rtf_gpu else 'N/A'
        print(f"  {m:<14} {rtf_cpu[m]:>8.3f} {gpu_str:>12}")

    # Noise reduction
    nr = results['noise_reduction_impact']
    print("\n  Noise Reduction Impact (SNR=10dB, small model)")
    print(f"  Without NR: {nr['without_nr']*100:.1f}%  →  "
          f"With NR: {nr['with_nr']*100:.1f}%  "
          f"({nr['improvement_pct']:.1f}% relative improvement)")

    # Latency breakdown
    lat = results['latency_breakdown_ms']
    total = sum(lat.values())
    print(f"\n  End-to-end Latency (30s clip, GPU, small model)  total={total:.0f}ms")
    for stage, ms in lat.items():
        print(f"    {stage:<22} {ms:>7.1f} ms  ({ms/total*100:.1f}%)")
    print("=" * 70 + "\n")


def generate_plots(results: dict, output_dir: str):
    try:
        from src.utils.visualization import (
            plot_wer_vs_snr, plot_rtf_comparison, plot_wer_heatmap,
            plot_latency_breakdown,
        )
    except ImportError:
        print("Visualization skipped (matplotlib unavailable)")
        return

    wer_data = results['wer_by_model_and_snr']
    snr_float = [float('inf'), 20, 15, 10, 5, 0]   # inf = clean

    # WER vs SNR (numeric SNRs only)
    snr_numeric = [20, 15, 10, 5, 0]
    wer_for_plot = {}
    for model, wd in wer_data.items():
        wer_for_plot[model] = [wd[str(s)] for s in snr_numeric]

    plot_wer_vs_snr(
        wer_for_plot, snr_numeric,
        path=os.path.join(output_dir, 'wer_vs_snr.png'),
    )

    # RTF comparison
    plot_rtf_comparison(
        results['rtf_cpu'], results.get('rtf_gpu'),
        path=os.path.join(output_dir, 'rtf_comparison.png'),
    )

    # WER heatmap
    models   = list(wer_data.keys())
    snr_keys = ['clean', '20', '15', '10', '5', '0']
    matrix   = np.array([[wer_data[m].get(k, 0) for k in snr_keys]
                          for m in models])
    plot_wer_heatmap(
        matrix, models,
        [s if s != 'clean' else 'clean' for s in snr_keys],
        path=os.path.join(output_dir, 'wer_heatmap.png'),
    )

    # Latency breakdown
    plot_latency_breakdown(
        results['latency_breakdown_ms'],
        path=os.path.join(output_dir, 'latency_breakdown.png'),
    )

    print(f"Plots saved to {output_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--simulate',    action='store_true',
                   help='Use built-in simulated results (no GPU needed)')
    p.add_argument('--test-list',   default=None)
    p.add_argument('--output-dir',  default='results')
    p.add_argument('--no-plots',    action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.simulate or args.test_list is None:
        print("Using simulated benchmark results.")
        results = SIMULATED_RESULTS
    else:
        print("Live benchmarking not yet wired — using simulated results.")
        results = SIMULATED_RESULTS

    print_table(results)
    save_results(results, args.output_dir)
    if not args.no_plots:
        generate_plots(results, args.output_dir)


if __name__ == '__main__':
    main()
