# PS-Detect-and-locate

Partial-spoof (PS) **detection & localization** on the
[PartialSpoof](https://github.com/nii-yamagishilab/partialspoof) database (v1.2),
using **DSP features + a light tabular model (LightGBM)** — no deep learning.

A partial spoof embeds short synthetic (TTS/VC) speech into otherwise genuine,
same-speaker audio. The task: decide whether an utterance contains spoof
(**detection**) and find which time spans are spoofed (**localization**).

## Approach (7-step pipeline)

```
[1] audio 16 kHz
[2] framing 25 ms / 10 ms hop
[3] DSP features:  magnitude (STFT, LFCC+Δ+ΔΔ) | phase (IF-dev, flux) |
                   discontinuity (spectral flux, log-energy d2) | seam (F0/novelty)
[4] window pooling 0.16 s (mean+std+max)  -> aligned 1:1 with segment labels
[5] classifier:  LogReg (baseline)  ->  LightGBM (main)
[6] window P(spoof):  localization = sequence,  detection = max pooling
[7] median smoothing + evaluation (Utterance EER / Range-EER)
```

**Why these features?** Spoof segments are the *same speaker* spliced in, so
speaker cues are useless. magnitude catches vocoder spectral texture; phase
catches vocoder phase incoherence; discontinuity/seam catch the concatenation
join. (Notably, F0/pitch is *not* useful — the construction's overlap-add already
smooths pitch at the seam; phase survives best.)

**Why LightGBM?** The pooled DSP features are 1101-dim *tabular* data, where
gradient-boosted trees give strong non-linear boundaries, stay light, and remain
interpretable — matching the assignment's "simple model" while avoiding deep nets.

## Dataset

PartialSpoof v1.2 (from Zenodo `5766198`). dev/train/eval used **in full** for the
official protocol. Verified facts:
- segment labels are `.npy` (per-frame 0/1, 7 resolutions); **1 = bonafide,
  0 = spoof**; `#frames = ceil(duration / resolution)`.
- dev = 24,844 = 2,548 bonafide (`LA_D_*`) + 22,296 spoof (`CON_D_*`).

> The dataset itself is **not** committed (see `.gitignore`); it lives in the local
> `partialspoof/` clone. Set `PS_DATA` to point the code elsewhere.

## Results (LightGBM, `full` feature set)

| protocol | Utt-EER (detection) | Range-EER (localization) |
|---|---:|---:|
| dev internal split (optimistic) | 0.84 % | 7.69 % |
| **train → dev** (in-domain) | 2.22 % | 7.93 % |
| **train → eval** (unseen attacks, final) | **13.88 %** | **24.60 %** |

The dev→eval gap is the known ASVspoof **unseen-attack generalization** challenge
(seen A01–A06 vs unseen A07–A19), not sample overfitting — speaker-leakage was
checked and ruled out. Full tables, methodology, and findings in
**[DESIGN.md](DESIGN.md)** and **[RESULTS.md](RESULTS.md)**.

## Quick start

```bash
pip install -r requirements.txt
# 1) download dev/train/eval into ./partialspoof (see partialspoof/01_download_database.sh)
python3 src/build_full.py train        # extract+cache features (multiprocessing)
python3 src/build_full.py dev
python3 src/run_full.py --test dev --backend lgbm     # official train->dev
python3 src/run_full.py --test eval --backend lgbm    # official train->eval
python3 src/run.py --backend lgbm                     # quick dev-internal sweep
python3 src/ps_data.py CON_D_0000000 0.16             # inspect one utterance
```

## Layout

```
src/        pipeline (ps_data, features, pipeline, model, evaluate, run, run_full,
            build_full, seam_detect, compare_methods) + make_report
analysis/   exploratory / figure scripts (label verification, seam DSP, ...)
figures/ results/ report/   artifacts
DESIGN.md   full design + results + reliability checks
```
