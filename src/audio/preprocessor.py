"""Audio preprocessing: noise reduction, normalization, pre-emphasis."""
import numpy as np


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to target_peak amplitude."""
    peak = np.abs(audio).max()
    if peak < 1e-7:
        return audio
    return audio * (target_peak / peak)


def rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-7:
        return audio
    return audio * (target_rms / rms)


def pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """High-pass pre-emphasis filter to amplify high frequencies."""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def spectral_subtraction(audio: np.ndarray, sample_rate: int,
                          n_fft: int = 512,
                          hop_length: int = 128,
                          noise_frames: int = 20,
                          alpha: float = 2.0,
                          beta: float = 0.002) -> np.ndarray:
    """
    Spectral subtraction noise reduction.

    Estimates noise from the first `noise_frames` frames (assumed to be
    background noise) and subtracts the noise spectrum from all frames.

    alpha: over-subtraction factor (higher = more aggressive)
    beta:  spectral floor (prevents musical noise)
    """
    window  = np.hanning(n_fft)
    frames  = _frame_signal(audio, n_fft, hop_length)
    spectra = np.fft.rfft(frames * window, n=n_fft)
    mag     = np.abs(spectra)
    phase   = np.angle(spectra)

    noise_est = mag[:noise_frames].mean(axis=0)

    mag_clean = np.maximum(mag - alpha * noise_est, beta * noise_est)
    spectra_clean = mag_clean * np.exp(1j * phase)
    frames_clean  = np.fft.irfft(spectra_clean, n=n_fft)

    return _overlap_add(frames_clean, hop_length, len(audio))


def add_noise(audio: np.ndarray, snr_db: float,
              noise: np.ndarray = None) -> np.ndarray:
    """Add white Gaussian noise at a given SNR (dB). Returns noisy audio."""
    signal_power = np.mean(audio ** 2)
    if noise is None:
        noise = np.random.randn(len(audio)).astype(np.float32)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return audio.copy()
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(audio + scaled_noise, -1.0, 1.0)


# ── Internal helpers ────────────────────────────────────────────────────

def _frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_len) // hop
    frames   = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        frames[i] = signal[i * hop: i * hop + frame_len]
    return frames


def _overlap_add(frames: np.ndarray, hop: int, signal_len: int) -> np.ndarray:
    frame_len = frames.shape[1]
    output    = np.zeros(signal_len, dtype=np.float32)
    window    = np.hanning(frame_len)
    norm      = np.zeros(signal_len, dtype=np.float32)
    for i, frame in enumerate(frames):
        start = i * hop
        end   = min(start + frame_len, signal_len)
        output[start:end] += (frame * window)[:end - start]
        norm[start:end]   += window[:end - start]
    mask         = norm > 1e-8
    output[mask] /= norm[mask]
    return output
