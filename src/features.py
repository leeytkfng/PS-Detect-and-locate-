#!/usr/bin/env python3
"""[2][3][4] Framing + DSP feature extraction + window pooling.

Frame analysis: 25 ms window / 10 ms hop @ 16 kHz (n_fft=512, win=400).

Feature GROUPS (each a per-frame sequence, then pooled to label windows):
  magnitude  -> STFT log-magnitude  (group "stft")
                LFCC + d + dd        (group "lfcc")
  phase      -> per-band IF-deviation + temporal phase flux   (group "phase")
  disc       -> per-band spectral flux + log-energy d1/d2     (group "disc")

Why: the spoof segment is the SAME speaker spliced in, so speaker cues are
useless. magnitude catches vocoder spectral texture; phase catches vocoder
phase incoherence; disc catches the concatenation-seam discontinuity. The seam
is sparse (a few frames), so we pool with mean + std + MAX -- max preserves the
peak discontinuity inside a window.
"""
import numpy as np
import librosa
from scipy.fftpack import dct

SR        = 16000
WIN       = 400          # 25 ms analysis window
N_FFT     = 512          # fft size (zero-padded)
HOP       = 160          # 10 ms hop
N_FILTERS = 40           # linear filterbank channels (LFCC)
N_CEPS    = 20           # cepstral coefficients
N_BANDS   = 16           # bands for phase/disc grouping
POOL_STATS = ("mean", "std", "max")

GROUPS = ("stft", "lfcc", "phase", "disc")


# ----------------------------------------------------------- shared STFT
def stft_complex(audio):
    return librosa.stft(audio, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                        window="hann", center=True)


# ----------------------------------------------------------- magnitude group
def _linear_filterbank(n_filters=N_FILTERS, n_fft=N_FFT, sr=SR):
    n_bins = n_fft // 2 + 1
    edges = np.linspace(0, sr / 2, n_filters + 2)
    bf = np.linspace(0, sr / 2, n_bins)
    fb = np.zeros((n_filters, n_bins), np.float32)
    for m in range(1, n_filters + 1):
        lo, ctr, hi = edges[m - 1], edges[m], edges[m + 1]
        fb[m - 1] = np.clip(np.minimum((bf - lo) / (ctr - lo),
                                       (hi - bf) / (hi - ctr)), 0, None)
    return fb


_FB = _linear_filterbank()


def lfcc_seq(audio):
    P = np.abs(stft_complex(audio)) ** 2
    logfbe = np.log(_FB @ P + 1e-10)
    return dct(logfbe, type=2, axis=0, norm="ortho")[:N_CEPS].T     # (T, N_CEPS)


def add_deltas(seq):
    x = seq.T
    w = max(3, min(9, x.shape[1] if x.shape[1] % 2 else x.shape[1] - 1))
    d1 = librosa.feature.delta(x, order=1, width=w)
    d2 = librosa.feature.delta(x, order=2, width=w)
    return np.concatenate([x, d1, d2], axis=0).T                   # (T, 3*N_CEPS)


def logmag_seq(audio):
    return np.log(np.abs(stft_complex(audio)) + 1e-10).T.astype(np.float32)


# ----------------------------------------------------------- band helpers
def _band_matrix(n_bands=N_BANDS, n_bins=N_FFT // 2 + 1):
    edges = np.linspace(0, n_bins, n_bands + 1).astype(int)
    M = np.zeros((n_bands, n_bins), np.float32)
    for b in range(n_bands):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        M[b, lo:hi] = 1.0 / (hi - lo)
    return M


_BM = _band_matrix()


def _princarg(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


# ----------------------------------------------------------- phase group
def phase_seq(audio):
    """(T, 2*N_BANDS): per-band magnitude-weighted IF-deviation + phase flux."""
    S = stft_complex(audio)
    mag = np.abs(S) + 1e-10
    phase = np.angle(S)
    n_bins = S.shape[0]
    dphi = np.diff(phase, axis=1)
    k = np.arange(n_bins)[:, None]
    expected = 2 * np.pi * HOP * k / N_FFT
    ifdev = np.abs(_princarg(dphi - expected))
    pflux = np.abs(_princarg(dphi))
    w = mag[:, 1:]
    ifd_b = _BM @ (ifdev * w) / (_BM @ w + 1e-10)
    pfl_b = _BM @ (pflux * w) / (_BM @ w + 1e-10)
    seq = np.concatenate([ifd_b, pfl_b], axis=0).T
    return np.vstack([seq[:1], seq]).astype(np.float32)


# ----------------------------------------------------------- disc group
def disc_seq(audio):
    """(T, N_BANDS+2): per-band spectral flux + |log-energy d1| + |d2|."""
    S = np.abs(stft_complex(audio))
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    flux_b = (_BM @ np.abs(np.diff(Sn, axis=1))).T
    flux_b = np.vstack([flux_b[:1], flux_b])
    loge = np.log((S ** 2).sum(axis=0) + 1e-10)
    d1 = np.gradient(loge); d2 = np.gradient(d1)
    return np.concatenate([flux_b, np.abs(d1)[:, None],
                           np.abs(d2)[:, None]], axis=1).astype(np.float32)


_SEQ_FN = {"stft": logmag_seq,
           "lfcc": lambda a: add_deltas(lfcc_seq(a)),
           "phase": phase_seq,
           "disc": disc_seq}


# ----------------------------------------------------------- [4] pooling
def pool_to_windows(seq, n_windows, hop=HOP, sr=SR, R=0.16, stats=POOL_STATS):
    """Pool a (T, d) frame sequence into (n_windows, len(stats)*d).

    Frame i (center=True) is at time i*hop/sr -> window floor(time/R).
    """
    T, d = seq.shape
    widx = np.floor(np.arange(T) * hop / sr / R).astype(int)
    out = np.zeros((n_windows, len(stats) * d), np.float32)
    for wi in range(n_windows):
        m = widx == wi
        if not m.any():
            if wi > 0:
                out[wi] = out[wi - 1]
            continue
        chunk = seq[m]
        parts = []
        for s in stats:
            parts.append(chunk.mean(0) if s == "mean" else
                         chunk.std(0) if s == "std" else chunk.max(0))
        out[wi] = np.concatenate(parts)
    return out


def group_features(audio, n_windows, R=0.16, groups=GROUPS, stats=POOL_STATS):
    """-> dict{group: (n_windows, *)} pooled features."""
    return {g: pool_to_windows(_SEQ_FN[g](audio), n_windows, R=R, stats=stats)
            for g in groups}


if __name__ == "__main__":
    import soundfile as sf, ps_data as P
    uid = "CON_D_0000000"
    a, _ = sf.read(f"{P.WAV}/{uid}.wav")
    n = len(P.load_seglab(0.16)[uid])
    for g, v in group_features(a.astype(np.float32), n).items():
        print(f"{g:6s} {v.shape}  nan={np.isnan(v).any()}")
