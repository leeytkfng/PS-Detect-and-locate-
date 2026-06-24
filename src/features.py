#!/usr/bin/env python3
"""DSP feature extraction for PartialSpoof spoof localization.

Three feature families (requested):
  1. LFCC                 - Linear-Frequency Cepstral Coefficients
  2. LFCC + d + dd        - LFCC with delta and delta-delta (temporal dynamics)
  3. STFT spectrogram     - log-magnitude STFT (raw spectral envelope)

Analysis is done on short frames (default 32 ms / 10 ms hop). To produce one
feature vector per *segment-label window* (resolution R, e.g. 0.16 s) we pool
the short-frame features inside each window with mean+std pooling, and align the
window count to the ground-truth segment labels.

Why these features?
  - LFCC uses a *linear* filterbank (unlike MFCC's mel), so it keeps high-
    frequency detail where vocoder/synthesis artifacts of spoofed speech live.
    It is the standard front-end for ASVspoof anti-spoofing baselines.
  - delta / delta-delta add how the cepstrum *changes* over time, which exposes
    the unnatural temporal smoothness of synthetic speech and the abrupt
    discontinuities at concatenation seams.
  - the raw log-STFT keeps the full spectro-temporal pattern (no cepstral
    compression) as a higher-dimensional comparison baseline.
"""
import numpy as np
import librosa
from scipy.fftpack import dct

SR        = 16000
N_FFT     = 512          # 32 ms @ 16 kHz
HOP       = 160          # 10 ms
N_FILTERS = 40           # linear filterbank channels
N_CEPS    = 20           # cepstral coefficients kept


# ---------------------------------------------------------------- filterbank
def _linear_filterbank(n_filters=N_FILTERS, n_fft=N_FFT, sr=SR):
    """Triangular filters equally spaced in Hz (linear), shape (n_filters, n_fft/2+1)."""
    n_bins = n_fft // 2 + 1
    f_max = sr / 2
    # n_filters+2 edge points equally spaced 0..f_max
    edges = np.linspace(0, f_max, n_filters + 2)
    bin_freqs = np.linspace(0, f_max, n_bins)
    fb = np.zeros((n_filters, n_bins), dtype=np.float32)
    for m in range(1, n_filters + 1):
        lo, ctr, hi = edges[m - 1], edges[m], edges[m + 1]
        left  = (bin_freqs - lo) / (ctr - lo)
        right = (hi - bin_freqs) / (hi - ctr)
        fb[m - 1] = np.clip(np.minimum(left, right), 0, None)
    return fb


_FB = _linear_filterbank()


# ---------------------------------------------------------------- base seqs
def _stft_power(audio):
    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, window="hann", center=True)
    return (np.abs(S) ** 2).astype(np.float32)          # (n_bins, T)


def lfcc_seq(audio):
    """-> (T, N_CEPS) linear-frequency cepstral coefficients."""
    P = _stft_power(audio)                                # (n_bins, T)
    fbe = _FB @ P                                         # (n_filters, T)
    logfbe = np.log(fbe + 1e-10)
    ceps = dct(logfbe, type=2, axis=0, norm="ortho")[:N_CEPS]   # (N_CEPS, T)
    return ceps.T                                         # (T, N_CEPS)


def add_deltas(seq):
    """(T, d) -> (T, 3d) : [feat, delta, delta-delta] along time."""
    x = seq.T                                             # (d, T)
    d1 = librosa.feature.delta(x, order=1, width=min(9, _odd(x.shape[1])))
    d2 = librosa.feature.delta(x, order=2, width=min(9, _odd(x.shape[1])))
    return np.concatenate([x, d1, d2], axis=0).T         # (T, 3d)


def logmag_seq(audio):
    """-> (T, n_bins) log-magnitude STFT spectrogram."""
    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, window="hann", center=True)
    return np.log(np.abs(S) + 1e-10).astype(np.float32).T  # (T, n_bins)


def _odd(n):
    """largest odd <= n and >=3 (librosa.delta needs odd width <= T)."""
    n = max(3, n)
    return n if n % 2 == 1 else n - 1


# ---------------------------------------------------------------- pooling
def pool_to_windows(seq, n_windows, hop=HOP, sr=SR, R=0.16):
    """Mean+std pool a (T, d) short-frame sequence into (n_windows, 2d).

    Frame i (center=True) is centered at time i*hop/sr; it belongs to window
    floor(time / R). Output is aligned/truncated to n_windows (the label count).
    """
    T, d = seq.shape
    times = np.arange(T) * hop / sr
    widx = np.floor(times / R).astype(int)
    out = np.zeros((n_windows, 2 * d), dtype=np.float32)
    for w in range(n_windows):
        m = widx == w
        if not m.any():                                  # empty tail window
            if w > 0:
                out[w] = out[w - 1]
            continue
        chunk = seq[m]
        out[w, :d]  = chunk.mean(axis=0)
        out[w, d:]  = chunk.std(axis=0)
    return out


def extract_all(audio, n_windows, R=0.16):
    """Return dict of the three pooled feature matrices, each (n_windows, *)."""
    lf = lfcc_seq(audio)
    return {
        "lfcc":     pool_to_windows(lf,             n_windows, R=R),
        "lfcc_dd":  pool_to_windows(add_deltas(lf), n_windows, R=R),
        "stft":     pool_to_windows(logmag_seq(audio), n_windows, R=R),
    }
