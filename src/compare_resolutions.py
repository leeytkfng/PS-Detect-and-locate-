#!/usr/bin/env python3
"""Compare segment labels across the 6 time resolutions, per utterance type.

Outputs:
  (A) a dataset-wide table: for bonafide / partial / full utterances, the mean
      spoof-frame ratio and mean #boundaries at each resolution.
  (B) one figure per type: spectrogram + stacked label strips (one per
      resolution), green=bonafide, red=spoof, so you can see how coarse vs fine
      resolution snaps the spoof boundaries.

Run:  python3 src/compare_resolutions.py
"""
import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import ps_data as P

RES = [0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]   # coarse -> fine
CMAP = ListedColormap(["#2ca02c", "#d62728"])       # 0=bonafide green, 1=spoof red


def utype(arr):
    s = set(arr.tolist())
    return "bonafide" if s == {"1"} else ("full" if s == {"0"} else "partial")


# ----------------------------------------------------------- (A) statistics
def stats_table():
    # use a single load per resolution; categorize by the 0.01 (finest) labels
    fine = P.load_seglab(0.01)
    types = {u: utype(a) for u, a in fine.items()}
    counts = {t: sum(v == t for v in types.values()) for t in ("bonafide", "partial", "full")}

    print("## (A) dataset-wide: mean spoof-frame ratio  [mean #boundaries]  per type\n")
    print(f"  utt counts: bonafide={counts['bonafide']}  "
          f"partial={counts['partial']}  full={counts['full']}\n")
    header = f"  {'res':>6} | " + " | ".join(f"{t:^22}" for t in
                                             ("bonafide", "partial", "full"))
    print(header); print("  " + "-" * (len(header) - 2))
    for r in RES:
        d = P.load_seglab(r)
        agg = {t: [[], []] for t in ("bonafide", "partial", "full")}  # [ratios],[bounds]
        for u, a in d.items():
            t = types[u]
            v = (a == "0").astype(int)        # 1 = spoof frame
            agg[t][0].append(v.mean())
            agg[t][1].append(int(np.abs(np.diff(v)).sum()))   # transitions
        cells = []
        for t in ("bonafide", "partial", "full"):
            ratio = np.mean(agg[t][0]) if agg[t][0] else float("nan")
            bnd = np.mean(agg[t][1]) if agg[t][1] else float("nan")
            cells.append(f"{ratio:6.3f}  [{bnd:5.2f}]")
        print(f"  {r:6.2f} | " + " | ".join(f"{c:^22}" for c in cells))
    print("\n  (ratio = fraction of frames labeled spoof; #boundaries = bonafide<->spoof switches)")


# ----------------------------------------------------------- (B) figure
def strip_figure(uid, out):
    audio, sr = sf.read(os.path.join(P.WAV, uid + ".wav"))
    if audio.ndim > 1:
        audio = audio[:, 0]
    dur = len(audio) / sr
    ulab = P.load_protocol()[uid][2]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(len(RES) + 1, 1, height_ratios=[5] + [1] * len(RES),
                          hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax0.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    ax0.set_ylabel("Hz")
    ax0.set_xlim(0, dur); ax0.set_xticks([])
    ax0.set_title(f"{uid}  (utt={ulab})   green=bonafide  red=spoof   "
                  f"strips: coarse(0.64s) -> fine(0.01s)")

    for i, r in enumerate(RES):
        ax = fig.add_subplot(gs[i + 1])
        arr = P.load_seglab(r)[uid]
        v = (arr == "0").astype(int)[None, :]      # 1=spoof
        ax.imshow(v, aspect="auto", cmap=CMAP, vmin=0, vmax=1,
                  extent=[0, len(arr) * r, 0, 1], interpolation="nearest")
        ax.set_yticks([]); ax.set_ylabel(f"{r:.2f}", rotation=0,
                                         ha="right", va="center", fontsize=8)
        ax.set_xlim(0, dur)
        if i < len(RES) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("time (s)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved:", out)


def pick_examples():
    """one representative utterance per type (partial: a clean single block)."""
    seg = P.load_seglab(0.16)
    ex = {}
    ex["bonafide"] = "LA_D_1024892"
    ex["full"] = next(u for u in seg if u.startswith("CON_D")
                      and set(seg[u].tolist()) == {"0"})
    ex["partial"] = "CON_D_0000000"
    return ex


if __name__ == "__main__":
    stats_table()
    print("\n## (B) per-type strip figures")
    figdir = os.path.join(P.REPO, "figures")
    os.makedirs(figdir, exist_ok=True)
    for t, uid in pick_examples().items():
        strip_figure(uid, os.path.join(figdir, f"reso_compare_{t}_{uid}.png"))
