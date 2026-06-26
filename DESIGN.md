# DESIGN — PartialSpoof DSP 탐지·국소화 설계 문서

> 한 줄 요약: **DSP 특징(크기/위상/불연속) + LightGBM**으로 부분 위조 음성을
> 탐지하고 위치까지 찾는다. 평가는 공식 **Utterance EER**(탐지) /
> **Range-EER**(국소화). 모델은 단순 로지스틱 회귀에서 **LightGBM으로 강화**됨.

---

## 1. 과제 개요
- **Partial Spoof(PS)**: 진짜 음성 발화 안에 TTS/VC로 만든 **짧은 가짜 구간**을 삽입·치환하는 공격. 단어/음절 하나로 의미를 뒤집음.
- **목표**: 디지털 신호처리(DSP)로 (a) **탐지**(발화에 가짜가 있나) + (b) **국소화**(어디가 가짜냐).
- **데이터**: PartialSpoof database v1.2 (ASVspoof2019 LA 기반).

## 2. 데이터
- **검증**: con_wav / protocols / segment_labels 실물 확인.
  - segment 라벨 = **.npy(프레임별 0/1, 6+1 해상도 0.01~0.64s)**, 규약 **1=진짜, 0=가짜**.
  - dev **24,844 = 진짜 2,548(LA_D) + 가짜 22,296(CON_D)**, 네 출처 개수 일치.
- **핵심 특성**: 가짜는 **같은 화자**의 구간을 VAD로 잘라 교차상관+overlap-add로 이어붙이고 −26dBov로 음량 정규화. → 화자/음량 단서 무력, **이음새+보코더 아티팩트**가 단서.

## 3. 파이프라인 (7단계)
```
[1] 오디오 16kHz
[2] 프레이밍 25ms 윈도우 / 10ms hop (n_fft=512, Hann)
[3] DSP 특징 (프레임별)        ← features.py
[4] 윈도우 풀링 0.16s, mean+std+max  ← 세그먼트 라벨과 1:1 정렬 (pipeline.py)
[5] 분류기  LogReg → LightGBM   ← model.py
[6] 윈도우 P(가짜): 국소화=시퀀스, 탐지=max 풀링
[7] median 평활 + 평가(Utt-EER / Range-EER)  ← model.py / evaluate.py
```
실행: `python3 src/run.py [--backend logreg|lgbm] [--smooth 5]`

## 4. DSP 특징 (무엇을, 왜)
| 그룹 | 내용 | 노리는 아티팩트 |
|---|---|---|
| **magnitude** | STFT 로그크기(257), LFCC+Δ+ΔΔ(60) | 보코더 스펙트럼 질감 (선형 필터뱅크=고주파/포먼트 보존) |
| **phase** | 순간주파수 편차 + 위상 flux (16밴드) | 보코더 위상 부정합 |
| **disc** | 스펙트럼 flux + 로그에너지 d1/d2 | 이음새 불연속 (음량 정규화에 강건) |
| **seam** | F0점프·유성전환·spectral novelty | 이음새(경계) 직격 |

풀링은 mean+std+**max** — 이음새는 sparse하므로 max가 윈도우 내 peak를 보존.

## 5. 모델 — 단순 회귀 → LightGBM (강화 완료)
- **베이스라인**: `LogisticRegression`(선형) = 단순 회귀. 가능성 확인용.
- **주모델**: **LightGBM**(Gradient Boosting, 비선형). `model.py`에 백엔드 2종을 두고 선택.
  ```python
  LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                 class_weight="balanced",            # 가짜/진짜 불균형 보정
                 subsample=0.8, colsample_bytree=0.8)  # 과적합 억제
  ```
- **사용 방식**: 윈도우 단위 DSP 특징(예: full=1101차원)으로 fit → `predict_proba`로 윈도우별 P(가짜). 탐지=max풀링, 후처리=median 평활(파이프라인 동일, 분류기만 교체).
- **추가 강화**: 시간맥락 ±1 이웃 윈도우 스택 → 국소화 추가 개선(아래 표).

## 6. 평가 지표 (공식 연동)
- **Utterance EER**(탐지): 표준 EER.
- **Range-EER**(국소화): 공식 `metric/RangeEER.py`와 **동일 라이브러리(pyannote DetectionCostFunction)·동일 로직** 재현. 입력만 그들의 model pkl 대신 우리 윈도우 점수, 정답은 더 고운 0.02s 라벨.

## 7. 실험 결과 (dev 1,500 샘플, 70/30 발화 분할 — *낙관적 기준선*)
| 특징 | 모델 | Utt-EER% | Range-EER% |
|---|---|---:|---:|
| stft | 단순회귀 | 9.56 | 13.96 |
| full | 단순회귀 | 11.07 | 11.05 |
| stft | LightGBM | 2.69 | 9.96 |
| stft+phase+disc | LightGBM | 1.85 | 8.27 |
| **full** | **LightGBM** | **0.84** | **7.69** |
| full + 시간맥락 k=1 | LightGBM | 1.51 | **6.96** |

> **단순회귀 → LightGBM 전환만으로 탐지 11%→0.8%, 국소화 11%→7.7%.**

## 8. 신뢰성 점검
### 8.1 화자 누수 (확인 완료 — 누수 아님)
| 분할 | 화자 겹침 | Utt-EER | Range-EER |
|---|---|---:|---:|
| 발화 분할(현재) | 20/20 전부 | 0.84 | 7.69 |
| 화자 분리 분할 | 0 | **0.59** | 8.61 |

화자를 완전 분리해도 탐지가 나빠지지 않음(오히려 0.59) → **모델이 화자를 외워 맞추는 게 아님**. PartialSpoof 같은 화자 설계로 화자 단서가 무력하다는 가설 확인. (dev 화자 20명 한계 → 진짜 검증은 train→eval.)

### 8.2 왜 수치가 높은가 / 정직한 프레이밍
- dev **내부 분할**(같은 분포)이라 낙관적. STFT 771차원 통째 등 과적합 여지.
- **정식 프로토콜(train 학습→dev/eval 평가)에선 수치가 오를(나빠질) 것이며 그게 신뢰 수치.**
- 발표 프레이밍: *"dev 내부분할 0.84% (낙관적) → 정식 eval에선 X% (신뢰 수치)"* 로 **둘 다 제시**.

## 9. 핵심 발견 (발표 스토리)
1. **STFT 탐지 최강 + 위상·불연속이 보탬**(직교 정보; 공식 Range-EER 13.96→11.05).
2. **반직관: F0(피치)는 안 통한다.** overlap-add가 "가장 매끄러운 join"을 골라 **피치 불연속을 적이 제거**. 반대로 **위상은 못 지워 경계검출 최강**(F1 0.355 vs F0 0.127).
3. **단순회귀 → LightGBM 도약**이 가장 큰 향상.
4. **공식 Range-EER 연동**으로 국소화를 제대로 측정.

## 10. 한계 & 남은 작업
- [진행중] **전체데이터 정식 평가** (train 학습 → dev/eval) — `build_full.py`(멀티프로세싱). 0.84%를 신뢰 수치로 확정하는 마지막 관문.
- [예정] F0 정리(안 통하니 제외, 단 "왜 빼나"는 발표에 유지), PDF/보고서 최신화, 비교표·15분 발표.

## 11. 코드 구조
```
src/  ps_data · features · pipeline · model · evaluate · run · seam_detect · build_full · make_report
analysis/  탐색·그림 (라벨검증, 해상도비교, 이음새 DSP, 탐지분해 등)
figures/ results/ report/  산출물
```
