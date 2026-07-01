# PartialSpoof 부분 위조 음성 — 탐지 · 국소화 (DSP + LightGBM)

[PartialSpoof](https://github.com/nii-yamagishilab/partialspoof) v1.2 데이터셋에서
**DSP 특징 + 경량 LightGBM**(딥러닝/SSL 미사용)으로 부분 위조 음성을
**탐지(가짜가 있나)** 하고 **국소화(어디가 가짜냐)** 한다.

> 핵심: 경량 handcrafted로 **in-domain 강한 성능** + **미학습 공격 한계를 정직·엄밀하게
> 측정·분석**. 통찰 "F0는 지워졌고 위상은 살아남았다", CQCC 5배 개선은 **shortcut으로 규명**.

## 핵심 결과 (full, LightGBM)
| 프로토콜 | Utterance EER(탐지) | Range-EER(국소화) |
|---|---:|---:|
| dev 내부분할 (낙관) | 0.84% | 7.69% |
| 정식 train→dev (in-domain) | 2.22% | 7.93% |
| **정식 train→eval (미학습, 최종)** | **13.88%** | **24.60%** |

신뢰 수치는 미학습 공격 **eval의 13.88% / 24.60%**. dev→eval 갭은 ASVspoof의 unseen-attack
일반화 한계(seen A01–06 vs unseen A07–19)이며, 화자 누수는 점검·배제됨.

## 문서 (docs/)
| 장 | 문서 |
|---|---|
| 1. 실험 개요 | [docs/01_실험개요.md](docs/01_실험개요.md) |
| 2. 실험 구성 | [docs/02_실험구성.md](docs/02_실험구성.md) |
| 3. 실험 데이터 | [docs/03_실험데이터.md](docs/03_실험데이터.md) |
| 4. 실험 대상 Feature | [docs/04_실험대상_feature.md](docs/04_실험대상_feature.md) |
| 5. 모델 도식화 | [docs/05_모델_도식화.md](docs/05_모델_도식화.md) |
| 6. 실험 유형들 | [docs/06_실험유형들.md](docs/06_실험유형들.md) |
| 7. 실험 결과 | [docs/07_실험결과.md](docs/07_실험결과.md) |
| 8. 한계점 및 향후 과제 | [docs/08_한계점_향후과제.md](docs/08_한계점_향후과제.md) |

(영문 요약 문서: [DESIGN.md](DESIGN.md), [RESULTS.md](RESULTS.md))

## 빠른 시작
```bash
pip install -r requirements.txt
# 1) partialspoof/ 에 dev/train/eval 다운로드 (partialspoof/01_download_database.sh)
python3 step3_dataset/build_full.py train      # 전체 특징추출 + 캐시(멀티프로세싱)
python3 step3_dataset/build_full.py dev
python3 step6_run/run_full.py --test dev  --backend lgbm   # 정식 train→dev
python3 step6_run/run_full.py --test eval --backend lgbm   # 정식 train→eval (최종)
python3 step6_run/run.py --backend lgbm                    # dev 내부분할(빠른 비교)
python3 step1_data/ps_data.py CON_D_0000000 0.16           # 한 발화 라벨 확인
```

## 저장소 구조 (7단계 파이프라인)
```
step1_data/      ps_data.py                 [1] 데이터 접근 + 라벨 검증
step2_features/  features.py, augment.py    [2][3] 프레이밍 + DSP 특징
step3_dataset/   pipeline.py, build_full.py [4] 윈도우 풀링 + 데이터셋
step4_model/     model.py                   [5][6] 분류기(6종) + 풀링/평활
step5_evaluate/  evaluate.py                [7] Utterance EER + Range-EER
step6_run/       run.py, run_full.py        실행(내부분할 / 정식 프로토콜)
step7_analysis/  분석·그림 스크립트
tools/           make_report.py             PDF 보고서 생성기
docs/ figures/ results/ report/            문서·산출물
```
> 데이터셋은 git 제외(`.gitignore`), `partialspoof/`에 위치. 환경변수 `PS_DATA`로 경로 지정.
