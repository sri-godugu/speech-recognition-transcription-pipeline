"""Visualization utilities for ASR evaluation and audio analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Dict, Optional


def plot_wer_vs_snr(results: Dict[str, List],
                    snr_levels: List[float],
                    title: str = 'WER vs SNR by Model Size',
                    path: str = None):
    """
    Line plot of WER (%) vs SNR (dB) for multiple model sizes.

    results: {'model_size': [wer_at_snr0, wer_at_snr1, ...]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ['o', 's', '^', 'D', 'v']
    for (model, wers), marker in zip(results.items(), markers):
        ax.plot(snr_levels, [w * 100 for w in wers],
                marker=marker, label=model, linewidth=2, markersize=6)
    ax.set_xlabel('SNR (dB)',  fontsize=12)
    ax.set_ylabel('WER (%)',   fontsize=12)
    ax.set_title(title,        fontsize=13)
    ax.legend(title='Model',   fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_rtf_comparison(rtf_cpu: Dict[str, float],
                         rtf_gpu: Dict[str, float] = None,
                         title: str = 'Real-Time Factor by Model Size',
                         path: str = None):
    """Grouped bar chart of RTF for CPU (and optionally GPU)."""
    models = list(rtf_cpu.keys())
    x      = np.arange(len(models))
    width  = 0.35 if rtf_gpu else 0.6

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_cpu = ax.bar(x - width/2 if rtf_gpu else x,
                      list(rtf_cpu.values()), width, label='CPU',
                      color='steelblue', alpha=0.85)
    if rtf_gpu:
        bars_gpu = ax.bar(x + width/2, list(rtf_gpu.values()), width,
                          label='GPU (A100)', color='coral', alpha=0.85)
        ax.legend(fontsize=10)

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2,
               label='Real-time (RTF=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('RTF (lower = faster)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars_cpu:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_wer_heatmap(wer_matrix: np.ndarray,
                      model_sizes: List[str],
                      snr_levels: List[float],
                      title: str = 'WER (%) — Model Size × SNR',
                      path: str = None):
    """Heatmap of WER across model sizes (rows) and SNR conditions (cols)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(wer_matrix * 100, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=30)
    plt.colorbar(im, ax=ax, label='WER (%)')
    ax.set_xticks(range(len(snr_levels)))
    ax.set_xticklabels([f'{s} dB' for s in snr_levels])
    ax.set_yticks(range(len(model_sizes)))
    ax.set_yticklabels(model_sizes)
    ax.set_xlabel('SNR', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(len(model_sizes)):
        for j in range(len(snr_levels)):
            ax.text(j, i, f'{wer_matrix[i,j]*100:.1f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if wer_matrix[i,j] > 0.15 else 'black')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                      n_fft: int = 512, hop: int = 128,
                      vad_segments=None,
                      title: str = 'Mel Spectrogram',
                      path: str = None):
    """Log mel spectrogram with optional VAD segment overlay."""
    import torch, torchaudio
    wav = torch.from_numpy(audio).unsqueeze(0)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=80
    )
    mel = mel_fn(wav).squeeze(0).numpy()
    mel = np.log1p(mel)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(mel, aspect='auto', origin='lower', interpolation='none',
              extent=[0, len(audio)/sample_rate, 0, 80])

    if vad_segments:
        for seg in vad_segments:
            ax.axvspan(seg.start_s, seg.end_s, alpha=0.25, color='lime',
                       label='speech')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Mel Bin',  fontsize=11)
    ax.set_title(title,       fontsize=12)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_latency_breakdown(latency_dict: Dict[str, float],
                            title: str = 'Latency Breakdown (ms)',
                            path: str = None):
    """Pie chart of latency stages."""
    stages = {k: v for k, v in latency_dict.items() if k != 'total_ms'}
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        list(stages.values()),
        labels=list(stages.keys()),
        autopct='%1.1f%%',
        startangle=140,
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    return fig
