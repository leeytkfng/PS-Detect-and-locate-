# Results — Partial-Spoof Detection & Localization

First-pass DSP baseline comparing three feature families on the PartialSpoof
**dev** split. Reproduce with:

```bash
python3 src/experiment.py            # full run (1500 utts, R=0.16s, seed=0)
python3 src/localize_demo.py stft    # localization figure
```

## 1. What features, and why

| feature | dim (mean+std) | what it captures | why for PS |
|---|---|---|---|
| **LFCC** | 40 | linear-filterbank cepstrum (20 ceps) | linear (not mel) scale keeps **high-freq** detail where vocoder/synthesis artifacts live; standard ASVspoof front-end |
| **LFCC + Δ + ΔΔ** | 120 | LFCC + 1st/2nd time derivatives | adds **temporal dynamics** — unnatural smoothness of synthetic speech and abrupt **concatenation seams** |
| **STFT spectrogram** | 514 | raw log-magnitude STFT (257 bins) | full spectro-temporal pattern with **no cepstral compression** (higher-dim upper baseline) |

Analysis frames: `n_fft=512` (32 ms), `hop=160` (10 ms). Short-frame features are
**mean+std pooled** into 0.16 s windows aligned 1:1 with the ground-truth segment
labels, so every window has both a feature vector and a 0/1 label.

## 2. How detection/localization works

- **Localization** = window-level binary classification (spoof vs bonafide).
  One `LogisticRegression` (`class_weight="balanced"`, standardized features) per
  feature family predicts `P(spoof)` for each 0.16 s window → a spoof probability
  curve over time.
- **Detection** = utterance score = **max** window `P(spoof)`; an utterance is
  flagged spoof if any window looks spoofed.
- **Split**: `GroupShuffleSplit` by **utterance** (70/30) → no utterance leaks
  between train and test. Threshold taken at the EER operating point.

## 3. Evaluation results

Dev sample: **1500 utts** (500 bonafide, 750 partial-spoof, 250 full-spoof) →
**32,800 windows**, spoof-window rate 0.441. Test = 30 % of utts, split by id.

| feature | dim | win-EER % | win-AUC | win-F1 | win-bAcc | utt-EER % | utt-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFCC | 40 | 22.36 | 0.842 | 0.756 | 0.776 | 21.31 | 0.878 |
| LFCC+Δ+ΔΔ | 120 | 21.21 | 0.861 | 0.768 | 0.788 | 20.64 | 0.881 |
| **STFT-spec** | 514 | **16.94** | **0.905** | **0.814** | **0.831** | **9.40** | **0.971** |

*win-\* = localization (per 0.16 s window); utt-\* = detection (per utterance).*

**Takeaways**
- All three features work well above chance (EER ≪ 50 %), confirming the labels
  and pipeline are sound.
- Δ+ΔΔ consistently improve over plain LFCC (temporal cues matter).
- The raw STFT spectrogram is the strongest here — utterance detection EER
  **9.4 %**, AUC **0.97** — at the cost of much higher dimensionality.

Localization example (held-out `CON_D_0000022`): predicted `P(spoof)` rises into
the decision region exactly over the ground-truth spoof span.
See [figures/localize_stft_CON_D_0000022.png](figures/localize_stft_CON_D_0000022.png).

## 3b. Is detection actually good? (breakdown)

The headline utt-EER could be inflated by *fully* spoofed utterances (trivial to
catch). Splitting by attack type (`python3 src/detect_breakdown.py`, test set:
146 bonafide / 221 partial / 83 full):

| feature | bona-vs-ALL EER% / AUC | bona-vs-FULL (easy) | bona-vs-PARTIAL (hard) |
|---|---|---|---|
| LFCC | 21.3 / 0.878 | 13.5 / 0.939 | 24.0 / 0.855 |
| LFCC+Δ+ΔΔ | 20.6 / 0.881 | 12.2 / 0.940 | 24.2 / 0.859 |
| **STFT-spec** | 9.4 / 0.971 | **5.8 / 0.990** | **10.7 / 0.964** |

**Verdict.** Yes, full-spoof pulls the headline down a bit — but the *realistic*
case (bonafide vs **partial** spoof) with STFT is still **EER 10.7 %, AUC 0.964**,
genuinely usable for a simple linear baseline. LFCC-family features are much
weaker on partial spoof (~24 % EER) → the **STFT spectrogram is doing the real
work**. For reference, deep SOTA reaches ~0.5–4 % utt-EER, so this is a solid
baseline, not a finished detector. **Localization (window EER 16.9 %) is the
weaker part and the main place to improve.**

## 4. Limitations / next steps

- Trains/tests on a **subsample of dev** split by utterance (the official `train`
  split is not downloaded yet). Numbers are a baseline, not the paper protocol.
- Linear classifier + mean/std pooling is deliberately simple. Likely gains:
  finer resolution (0.02 s), GMM/LightGBM/CNN back-ends, score smoothing for
  cleaner spans, and proper train→dev→eval evaluation.
