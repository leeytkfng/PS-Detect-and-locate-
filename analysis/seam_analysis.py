#!/usr/bin/env python3
"""삽입된 TTS/VC 구간의 '어색함'이 고전 DSP로, 특히 이음새에서 측정 가능함을 보인다.

발화 하나에 대해 프레임 단위(10 ms)로 계산:
  - 단시간 에너지 (RMS)
  - 영교차율 (ZCR)
  - 스펙트럼 중심 (spectral centroid)
  - 스펙트럼 flux (프레임간 변화 -> 불연속에서 튐)
그리고 정답 가짜 구간을 겹쳐 그린다. 이음새(진짜<->가짜 전환)를 표시하며,
거기서 flux가 튀고, 가짜 구간의 질감(flux/ZCR/centroid)이 진짜와 다른 경향.

실행:  python3 analysis/seam_analysis.py [uid] [해상도]
"""
import os, sys
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import ps_data as P

N_FFT, HOP = 512, 160      # 32 ms / 10 ms @ 16 kHz


def dsp_curves(audio, sr):
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))     # (bins, T)
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT,
                                             hop_length=HOP)[0]
    cen = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)                  # 프레임별 정규화
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
    seams = [s for s, _, _ in intervals[1:]]      # 내부 경계(이음새)

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

    # 간단 수치 대비: 가짜 vs 진짜 프레임 평균
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
