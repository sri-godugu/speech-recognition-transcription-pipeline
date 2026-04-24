"""Audio loading and resampling utilities."""
import os
import numpy as np
import torch
import torchaudio

WHISPER_SR = 16_000   # Whisper expects 16 kHz mono float32


def load_audio(path: str, target_sr: int = WHISPER_SR) -> np.ndarray:
    """Load any audio file and return mono float32 array at target_sr."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def load_audio_segment(path: str, start_s: float, end_s: float,
                       target_sr: int = WHISPER_SR) -> np.ndarray:
    """Load a specific time segment from an audio file."""
    wav, sr = torchaudio.load(path,
                               frame_offset=int(start_s * sr),
                               num_frames=int((end_s - start_s) * sr))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def get_audio_info(path: str) -> dict:
    """Return metadata for an audio file without loading the full signal."""
    info = torchaudio.info(path)
    return {
        'sample_rate':  info.sample_rate,
        'num_frames':   info.num_frames,
        'num_channels': info.num_channels,
        'duration_s':   info.num_frames / info.sample_rate,
        'encoding':     info.encoding,
    }


def audio_from_tensor(wav: torch.Tensor, sr: int,
                      target_sr: int = WHISPER_SR) -> np.ndarray:
    """Convert a torch.Tensor waveform to Whisper-ready numpy array."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)
