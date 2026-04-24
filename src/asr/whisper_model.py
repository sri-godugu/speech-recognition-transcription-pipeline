"""Whisper model wrapper with lazy loading, device management, and fp16 support."""
from __future__ import annotations
import time
import torch
import numpy as np

SUPPORTED_MODELS = ('tiny', 'base', 'small', 'medium',
                    'large', 'large-v2', 'large-v3')


class WhisperModel:
    """
    Thin wrapper around openai-whisper.

    Lazy-loads the model on first use; caches across calls.
    Uses fp16 automatically on CUDA, fp32 on CPU.
    """

    def __init__(self, model_size: str = 'small', device: str = None,
                 download_root: str = None):
        if model_size not in SUPPORTED_MODELS:
            raise ValueError(f"model_size must be one of {SUPPORTED_MODELS}")
        self.model_size    = model_size
        self.device        = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.download_root = download_root
        self._model        = None

    def load(self):
        if self._model is not None:
            return
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
        self._model = whisper.load_model(
            self.model_size,
            device=self.device,
            download_root=self.download_root,
        )
        if self.device == 'cuda' and torch.cuda.is_available():
            self._model = self._model.half()

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def transcribe(self, audio: np.ndarray,
                   language: str = None,
                   task: str = 'transcribe',
                   word_timestamps: bool = True,
                   beam_size: int = 5,
                   best_of: int = 5,
                   temperature: float = 0.0,
                   **kwargs) -> dict:
        """
        Transcribe a float32 numpy array (16 kHz, mono).

        Returns Whisper's native result dict with:
          - 'text':     full transcript string
          - 'segments': list of {id, start, end, text, words}
          - 'language': detected language code
        """
        t0 = time.perf_counter()
        result = self.model.transcribe(
            audio,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            **kwargs,
        )
        result['_inference_time_s'] = time.perf_counter() - t0
        return result

    def detect_language(self, audio: np.ndarray) -> tuple[str, dict]:
        """Detect spoken language from the first 30 seconds of audio."""
        import whisper
        mel  = whisper.log_mel_spectrogram(audio[:16000 * 30]).to(self.device)
        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)
        return lang, probs

    def __repr__(self):
        loaded = 'loaded' if self._model is not None else 'not loaded'
        return f"WhisperModel(size={self.model_size}, device={self.device}, {loaded})"
