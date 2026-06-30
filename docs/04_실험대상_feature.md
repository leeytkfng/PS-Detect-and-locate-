# 4. 실험 대상 Feature

같은 화자 splice라 **화자 단서는 무의미** → 이음새·보코더 아티팩트만 노린다.
공통 프레이밍: 25ms 윈도우 / 10ms hop, n_fft=512, Hann. 표기: \(X_t[k]\) = STFT.

## 채택 특징 (4그룹, full = 1101-d)
| 그룹 | 정의(요약) | 차원 | 노리는 것 |
|---|---|---:|---|
| **STFT** | log-magnitude `log|Xₜ[k]|` | 771 | 보코더 스펙트럼 질감(통째) |
| **LFCC** | `|X|² → linear FB(40) → log → DCT-II(20) +Δ+ΔΔ` | 180 | mel 아닌 **linear**=고주파 보존 |
| **phase** | IFD + phase flux (mag-weighted, band-pooled) | 96 | 보코더 위상 부정합 |
| **disc** | normalized spectral flux + 로그에너지 d1·d2 | 54 | 이음새 불연속 |

### 수식
- **LFCC**: `cₜ[i] = Σₘ log( Σₖ Hₘ[k]·|Xₜ[k]|² )·cos[ πi/M (m+½) ]`
  (`Hₘ` = 선형 삼각 필터뱅크 40개, DCT-II 20계수 + Δ + ΔΔ)
- **phase (IFD)**: `IFDₜ[k] = | princ_arg( ∠Xₜ[k] − ∠Xₜ₋₁[k] − 2πk·hop/N ) |`
  (princ_arg = 위상 wrap, `2πk·hop/N` = 기대 위상 전진), 크기 가중 후 16밴드 풀링.
- **disc**: `dₜ = Σₖ |S̃ₜ[k] − S̃ₜ₋₁[k]|`, `S̃ₜ[k]=|Xₜ[k]|/Σₖ|Xₜ[k]|`
  (프레임 정규화 스펙트럼 flux) + `|d/dt|, |d²/dt²| log-energy`.

### 윈도우 풀링
각 프레임 시퀀스 `z`를 0.16s 윈도우 `W`로:
`φ(W) = [ meanₜ z, stdₜ z, maxₜ z ]` — **max가 sparse한 이음새 peak를 보존**.
최종 입력 = concat(stft, lfcc, phase, disc) = **1101-d** → LightGBM.

## 검토했으나 미채택 (표준특징 비교)
| 특징 | 결과 | 판정 |
|---|---|---|
| **CQCC**(상수-Q 켑스트럼) | 단독 eval 3.29%(STFT급), full+cqcc eval 2.90% | **shortcut**(9장) — 순수 기여 ~1.5pt, 기각 |
| **고주파 대역**(>4/6kHz 비율, rolloff, flatness) | 단독 약함 | STFT가 이미 커버, 미채택 |
| **MGD**(수정 그룹지연) | 경계검출 F1 0.331 (phase 0.355) | phase와 비등~열위, 융합도 미미, 미채택 |

→ "왜 표준특징 안 썼나" 방어: **테스트했고, 기존 magnitude/phase와 중복 또는 shortcut**.
