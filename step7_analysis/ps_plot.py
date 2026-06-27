#!/usr/bin/env python3
"""PartialSpoof 발화의 파형 + 스펙트로그램을 그리고, (segment_labels의) 가짜
구간을 음영 처리하여 오디오<->라벨 매칭을 눈으로 확인한다."""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import os, sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ps_data as P


def plot(uid, res=0.16, out=None):
    audio, sr = sf.read(os.path.join(P.WAV, uid + ".wav"))
    if audio.ndim > 1:
        audio = audio[:, 0]
    dur = len(audio) / sr
    frames = P.load_seglab(res)[uid].tolist()
    intervals = P.frames_to_intervals(frames, res)
    spk, sysid, ulab = P.load_protocol()[uid]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(audio)) / sr

    ax1.plot(t, audio, lw=0.4, color="black")
    ax1.set_ylabel("amplitude")
    ax1.set_title(f"{uid}   utt_label={ulab}  (speaker={spk}, system={sysid})   "
                  f"red = spoof segments @ {res}s res")

    ax2.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    ax2.set_ylabel("freq (Hz)")
    ax2.set_xlabel("time (s)")

    for s, e, lab in intervals:
        if lab == "spoof":
            for ax in (ax1, ax2):
                ax.axvspan(s, min(e, dur), color="red", alpha=0.25, zorder=5)
    for ax in (ax1, ax2):
        ax.set_xlim(0, dur)

    fig.tight_layout()
    figdir = os.path.join(P.REPO, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = out or os.path.join(figdir, f"verify_{uid}_{res}.png")
    fig.savefig(out, dpi=110)
    print("saved:", out)
    print("intervals:")
    for s, e, lab in intervals:
        print(f"   {s:6.2f}-{e:6.2f}s  {lab}")
    return out


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "CON_D_0000000"
    res = float(sys.argv[2]) if len(sys.argv) > 2 else 0.16
    plot(uid, res)
