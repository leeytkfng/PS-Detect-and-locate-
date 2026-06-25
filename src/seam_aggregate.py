#!/usr/bin/env python3
"""Aggregate DSP-cue contrast (bonafide vs spoof frames) over many partial-spoof
utterances, so the seam/segment 'awkwardness' is shown statistically, not just
on a single example.

For N sampled partial-spoof utterances we compute per 10 ms frame:
  RMS energy, zero-crossing rate, spectral centroid, spectral flux,
then average each measure separately over bonafide frames and spoof frames.

Run:  python3 src/seam_aggregate.py [N] [resolution]
"""
import os, sys
import numpy as np
import soundfile as sf
import librosa

import ps_data as P
import dataset as D

N_FFT, HOP = 512, 160
MEASURES = ["rms", "zcr", "centroid", "flux"]


def frame_curves(audio, sr):
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT,
                                             hop_length=HOP)[0]
    cen = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    flux = np.concatenate([[0.0], np.sqrt((np.diff(Sn, axis=1) ** 2).sum(axis=0))])
    t = np.arange(S.shape[1]) * HOP / sr
    return t, {"rms": rms, "zcr": zcr, "centroid": cen, "flux": flux}


def run(n=300, R=0.02, seed=1):
    uids = D.sample_uids(R, 0, n, 0, seed=seed)        # n partial-spoof utts
    seg = P.load_seglab(R)
    acc = {k: [[], []] for k in MEASURES}              # [bonafide], [spoof]

    for u in uids:
        a, sr = sf.read(os.path.join(P.WAV, u + ".wav"))
        if a.ndim > 1:
            a = a[:, 0]
        a = a.astype(np.float32)
        t, cur = frame_curves(a, sr)
        is_spoof = np.zeros(len(t), bool)
        for s, e, lab in P.frames_to_intervals(seg[u].tolist(), R):
            if lab == "spoof":
                is_spoof |= (t >= s) & (t < e)
        m = min(len(t), *(len(v) for v in cur.values()))
        is_spoof = is_spoof[:m]
        for k in MEASURES:
            v = cur[k][:m]
            acc[k][0].append(v[~is_spoof])
            acc[k][1].append(v[is_spoof])

    frame_ms = int(HOP / sr * 1000)
    print(f"# DSP-cue contrast over {len(uids)} partial-spoof utts "
          f"(R={R}s, {frame_ms} ms frames)")
    print(f"  {'measure':<12}{'bonafide':>11}{'spoof':>11}{'delta%':>9}")
    rows = []
    for k in MEASURES:
        b = np.concatenate(acc[k][0]).mean()
        s = np.concatenate(acc[k][1]).mean()
        rows.append((k, b, s, (s - b) / b * 100))
        print(f"  {k:<12}{b:>11.3f}{s:>11.3f}{(s - b) / b * 100:>8.1f}%")
    print("\n  (spoof frames vs bonafide frames; flux up + energy down "
          "= the 'awkwardness' signal)")
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    run(n, R)
