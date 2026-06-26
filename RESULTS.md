# Results — Partial-Spoof Detection & Localization

DSP features + LightGBM on PartialSpoof. Metrics: **Utterance EER** (detection),
**Range-EER** (localization, official pyannote-based). See **[DESIGN.md](DESIGN.md)**
for the full method.

## 1. Headline (official protocol, `full` features, LightGBM)

| protocol | Utt-EER | Range-EER |
|---|---:|---:|
| dev internal split (optimistic) | 0.84 % | 7.69 % |
| **train → dev** (in-domain) | 2.22 % | 7.93 % |
| **train → eval** (unseen attacks, **final**) | **13.88 %** | **24.60 %** |

The three-stage drop is the honest story: optimistic → in-domain → unseen-attack
held-out. eval uses unseen TTS/VC (ASVspoof A07–A19), so the gap is expected.

## 2. Feature ablation (official train → eval, LightGBM)

| feature set | dim | win-EER | Utt-EER | Range-EER |
|---|---:|---:|---:|---:|
| lfcc | 180 | 24.46 | 19.27 | 25.47 |
| stft | 771 | 22.99 | 15.61 | 25.22 |
| stft+phase+disc | 921 | 22.96 | 15.83 | 26.02 |
| **full** | 1101 | 21.57 | **13.88** | **24.60** |

`full` (magnitude + phase + discontinuity) is best across detection and
localization. On the in-domain train→dev split the ordering is the same and the
phase/disc additions help more clearly (Range-EER 9.78 → 8.10 → 7.93).

## 3. Model: simple regression → LightGBM

| feature | model | Utt-EER (train→dev) | Range-EER |
|---|---|---:|---:|
| full | Logistic Regression | 12.32 | 12.81 |
| full | **LightGBM** | **2.22** | **7.93** |

Swapping the classifier (same features) is the single biggest gain. Tabular
method comparison is in §6.

## 4. DSP cue analysis (300 partial-spoof utts, spoof vs bonafide frames)

| measure | bonafide | spoof | Δ |
|---|---:|---:|---:|
| RMS energy | 0.024 | 0.015 | −36.7 % |
| zero-crossing rate | 0.147 | 0.120 | −18.1 % |
| spectral centroid (Hz) | 1810 | 1708 | −5.7 % |
| spectral flux | 0.077 | 0.082 | +6.9 % |

Cues exist but no single hand measure separates cleanly (5–37 %, direction varies
per utterance) — hence a full feature vector + classifier, not a threshold.

## 5. Seam (boundary) detection (300 partial utts, ±50 ms, P/R/F1)

| cue | F1 |
|---|---:|
| **phase** | **0.355 (best)** |
| spectral novelty | 0.301 |
| F0 (robust) | 0.127 (worst) |

**Counter-intuitive finding:** F0/pitch is the *weakest* boundary cue. The
construction joins segments with cross-correlation + overlap-add at the smoothest
point, so the attacker effectively removed the pitch discontinuity; phase
incoherence cannot be smoothed away and survives best. Seam features help
localization (Range-EER, boundary-sensitive) but not window-level region
classification (the seam is too sparse).

## 6. Tabular method comparison

Same `full` features, official train→dev, classifier swapped (Range-EER on a
1,044-utt subset):

| method | win-EER | Utt-EER | Utt-AUC | Range-EER | train (s) |
|---|---:|---:|---:|---:|---:|
| logreg (linear) | 15.26 | 12.32 | 0.947 | 12.81 | 105 |
| rf (bagging trees) | 11.64 | 4.43 | 0.987 | 11.59 | 93 |
| mlp (shallow net) | 7.48 | 3.02 | 0.993 | **7.20** | 364 |
| histgb (sklearn boosting) | 7.69 | 2.24 | 0.996 | 8.33 | 57 |
| **lgbm (LightGBM)** | 7.67 | **2.22** | **0.996** | 8.04 | **30** |

**LightGBM is the best choice**: best detection (2.22 %), near-best localization,
and **fastest (30 s, ~12× faster than the MLP)**. Both boosting-tree methods
(lgbm/histgb) clearly beat linear (logreg) and bagging (rf), confirming
gradient boosting is the right family for these tabular DSP features. The MLP wins
localization marginally but costs far more and loses on detection.

## 7. Reliability

- **Speaker leakage**: speaker-disjoint split does not hurt detection
  (0.84 → 0.59) → not memorizing speakers (PS is same-speaker by design).
- **Gap nature**: not sample overfitting (dev is fine at 2.22 %); it is
  specialization to seen attacks (A01–A06) failing on unseen ones (A07–A19).

## 8. Limitations / next

- Closing the unseen-attack gap needs SSL features / augmentation / domain
  generalization — out of scope for a "simple DSP model". The gap is reported
  honestly as a finding.
- Reproduce: `python3 src/run_full.py --test eval --backend lgbm`.
