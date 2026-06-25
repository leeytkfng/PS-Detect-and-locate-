#!/usr/bin/env python3
"""Show that the 'awkwardness' of an inserted TTS/VC segment is measurable with
classic DSP, especially at the concatenation seam.

For one utterance we compute, frame by frame (10 ms):
  - short-time energy (RMS)
  - zero-crossing rate (ZCR)
  - spectral centroid
  - spectral flux  (frame-to-frame spectral change -> spikes at discontinuities)
and overlay the ground-truth spoof span. Seams (bonafide<->spoof switches) are
marked; spectral flux there should jump, and the spoof region's texture (flux/
ZCR/centroid) tends to differ from the bonafide part.

Run:  python3 src/seam_analysis.py [uid] [resolution]
"""
import os, sys
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ps_data as P

N_FFT, HOP = 512, 160      # 32 ms / 10 ms @ 16 kHz


def dsp_curves(audio, sr):
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))     # (bins, T)
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT,
                                             hop_length=HOP)[0]
    cen = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)                  # normalize per frame
    flux = np.sqrt(((np.diff(Sn, axis=1)) ** 2).sum(axis=0))        # (T-1,)
    flux = np.concatenate([[0.0], flux])
    t = np.arange(S.shape[1]) * HOP / sr
    return t, rms, zcr, cen, flux


def main(uid="CON_D_0000000", R=0.02):
    audio, sr = sf.read(os.path.join(P.WAV, uid + ".wav"))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    dur = len(audio) / sr
    labs = P.load_seglab(R)[uid].tolist()
    intervals = P.frames_to_intervals(labs, R)
    seams = [s for s, _, _ in intervals[1:]]      # interior boundaries

    t, rms, zcr, cen, flux = dsp_curves(audio, sr)

    panels = [("short-time energy (RMS)", rms),
              ("zero-crossing rate", zcr),
              ("spectral centroid (Hz)", cen),
              ("spectral flux (discontinuity)", flux)]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(12, 9), sharex=True)

    ax0 = axes[0]
    ax0.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    ax0.set_ylabel("Hz")
    ax0.set_title(f"{uid}  -  DSP cues vs spoof label (R={R}s)   "
                  f"red span=spoof, dashed=seam")

    for ax, (name, y) in zip(axes[1:], panels):
        ax.plot(t, y, color="C0", lw=0.8)
        ax.set_ylabel(name, fontsize=8)

    for ax in axes:
        for s, e, lab in intervals:
            if lab == "spoof":
                ax.axvspan(s, min(e, dur), color="red", alpha=0.15)
        for x in seams:
            ax.axvline(x, color="k", ls="--", lw=0.8)
        ax.set_xlim(0, dur)
    axes[-1].set_xlabel("time (s)")

    # quick numeric contrast: spoof vs bonafide frame means
    fr_t = t
    is_spoof = np.zeros(len(fr_t), bool)
    for s, e, lab in intervals:
        if lab == "spoof":
            is_spoof |= (fr_t >= s) & (fr_t < e)
    print(f"# {uid}  (R={R}s)  seams at: " +
          ", ".join(f"{x:.2f}s" for x in seams))
    print(f"  {'measure':<22}{'bonafide':>10}{'spoof':>10}")
    for name, y in panels:
        b = y[~is_spoof].mean() if (~is_spoof).any() else float('nan')
        s = y[is_spoof].mean() if is_spoof.any() else float('nan')
        print(f"  {name:<22}{b:>10.3f}{s:>10.3f}")

    figdir = os.path.join(P.REPO, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"seam_{uid}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("saved:", out)
    return out


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "CON_D_0000000"
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    main(uid, R)
