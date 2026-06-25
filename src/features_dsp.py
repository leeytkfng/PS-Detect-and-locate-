#!/usr/bin/env python3
"""Targeted DSP features for PartialSpoof, aimed at the two cues that the
same-speaker splice construction leaves behind:

  (1) the concatenation SEAM  -> discontinuity features
  (2) the inserted SYNTHETIC segment -> phase artifacts

Speaker identity is useless here (spoof segment is the SAME speaker), so we look
at how the signal *moves* (phase/energy continuity) and at phase coherence,
which vocoders reproduce poorly.

Feature families (all pooled mean+std into segment-label windows, like features.py):

  phase : per-band instantaneous-frequency deviation + temporal phase flux
          -> vocoder phase incoherence + seam phase jumps
  disc  : per-band spectral flux + log-energy 1st/2nd derivative
          -> spectral/energy discontinuity at the join
  stft_pd : STFT log-magnitude  ++  phase  ++  disc   (does phase ADD to STFT?)

Reuses STFT config + pooling from features.py.
"""
import numpy as np
import librosa

import features as F   # N_FFT, HOP, SR, pool_to_windows, logmag_seq

N_BANDS = 16           # group the 257 STFT bins into a few linear bands


def _band_matrix(n_bands=N_BANDS, n_bins=F.N_FFT // 2 + 1):
    """Simple non-overlapping linear band-averaging matrix (n_bands, n_bins)."""
    edges = np.linspace(0, n_bins, n_bands + 1).astype(int)
    M = np.zeros((n_bands, n_bins), dtype=np.float32)
    for b in range(n_bands):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        M[b, lo:hi] = 1.0 / (hi - lo)
    return M


_BM = _band_matrix()


def _princarg(x):
    """wrap phase to (-pi, pi]."""
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def _complex_stft(audio):
    return librosa.stft(audio, n_fft=F.N_FFT, hop_length=F.HOP,
                        window="hann", center=True)


def phase_seq(audio):
    """-> (T, 2*N_BANDS): per-band IF-deviation + per-band temporal phase flux."""
    S = _complex_stft(audio)
    mag = np.abs(S) + 1e-10
    phase = np.angle(S)
    n_bins, T = S.shape

    # temporal phase difference, minus expected linear advance per bin
    dphi = np.diff(phase, axis=1)                          # (bins, T-1)
    k = np.arange(n_bins)[:, None]
    expected = 2 * np.pi * F.HOP * k / F.N_FFT
    ifdev = np.abs(_princarg(dphi - expected))            # IF deviation (bins, T-1)
    pflux = np.abs(_princarg(dphi))                       # raw phase flux

    # magnitude-weight so silent bins don't dominate, then band-average
    w = mag[:, 1:]
    ifd_b = _BM @ (ifdev * w) / (_BM @ w + 1e-10)         # (n_bands, T-1)
    pfl_b = _BM @ (pflux * w) / (_BM @ w + 1e-10)
    seq = np.concatenate([ifd_b, pfl_b], axis=0).T        # (T-1, 2*N_BANDS)
    seq = np.vstack([seq[:1], seq])                       # pad to T
    return seq.astype(np.float32)


def disc_seq(audio):
    """-> (T, N_BANDS+2): per-band spectral flux + log-energy 1st/2nd derivative."""
    S = np.abs(_complex_stft(audio))
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    flux = np.abs(np.diff(Sn, axis=1))                    # (bins, T-1)
    flux_b = (_BM @ flux).T                               # (T-1, n_bands)
    flux_b = np.vstack([flux_b[:1], flux_b])

    loge = np.log((S ** 2).sum(axis=0) + 1e-10)           # (T,)
    d1 = np.gradient(loge)
    d2 = np.gradient(d1)
    seq = np.concatenate([flux_b, np.abs(d1)[:, None], np.abs(d2)[:, None]], axis=1)
    return seq.astype(np.float32)


def extract_dsp(audio, n_windows, R=0.16):
    """pooled (n_windows, *) for phase / disc / stft_pd."""
    ph = F.pool_to_windows(phase_seq(audio), n_windows, R=R)
    di = F.pool_to_windows(disc_seq(audio),  n_windows, R=R)
    st = F.pool_to_windows(F.logmag_seq(audio), n_windows, R=R)
    return {
        "phase":   ph,
        "disc":    di,
        "stft_pd": np.concatenate([st, ph, di], axis=1),
    }


if __name__ == "__main__":
    import soundfile as sf, ps_data as P
    uid = "CON_D_0000000"
    a, sr = sf.read(f"{P.WAV}/{uid}.wav")
    n = len(P.load_seglab(0.16)[uid])
    feats = extract_dsp(a.astype(np.float32), n, R=0.16)
    for k, v in feats.items():
        print(f"{k:8s} {v.shape}  nan={np.isnan(v).any()}")
