# 벤치마크 프로토콜 (고정 — 이후 변경 금지)

목적: 공개된 대표 모델들을 우리 코퍼스 전부에 **동일 조건**으로 적용해 상대우위를 측정한다.

## 공통 프로토콜
- 입력 길이: 모든 시퀀스를 T=256 으로 선형 보간 리샘플
- 정규화: 모달별 per-sample z-score (표준편차 0 은 1.0 으로 치환 — epsilon 누설 방지)
- 모델 선택: **valid AUC 최대 epoch** 의 test 예측 사용
- 학습: Adam lr=7e-4 wd=1e-4, batch 64, epoch 60, BCEWithLogits(pos_weight=클래스비)
- 시드: 기본 5 (0~4)
- 보고: **(a) seed 평균±표준편차, (b) seed 앙상블 — 둘 다**
- 임계값: 0.5 고정 (튜닝 금지)
- 지표: Accuracy, Precision, Recall, F1, AUC, #params

## 코퍼스별 split
| 코퍼스 | split |
|---|---|
| D-Vlog | labels.csv 공식 fold (train 639 / valid 101 / test 212) |
| LMVD | 공식 split 없음 → 5-fold stratified CV, train 의 15% 를 valid, out-of-fold 예측으로 지표 산출 |
| CMDC | n=44 → 5-fold stratified CV |
| E-DAIC | AVEC2019 공식 분할 (없으면 5-fold CV) |
| MUD3 | CCAC-baseline 제공 split. raw / aligned 두 조건 |

## 피처 (공개 논문과 동일)
| 코퍼스 | visual | acoustic |
|---|---|---|
| D-Vlog | 68 landmark × 2 = 136 (dlib) | 25 LLD (OpenSmile) |
| LMVD | OpenFace (AU + gaze + pose) | VGGish 128 |
| CMDC | OpenFace AU | — |
| E-DAIC | OpenFace AU | — |
| MUD3 | CCAC 제공 | CCAC 제공 |

## 모델
| 이름 | 출처 | 비고 |
|---|---|---|
| `blstm` | 2019 baseline | 모달별 BiLSTM + concat |
| `tfn` | Zadeh et al., EMNLP 2017 | Tensor Fusion (외적) |
| `depdetector` | Yoon et al., AAAI 2022 | **D-Vlog 원논문** — Transformer + cross-modal attention |
| `tamfn` | Zhou et al., IEEE TNSRE 2023 | GTCN + IFE + TAMF |
| `ours` | 제안 | 윈도우 coordination → SPD → GRU + 음성 1D-CNN |

## 외부 공개 F1 (CAF-Mamba, arXiv 2601.21648 Table 2) — 재현 목표
| 모델 | D-Vlog | LMVD |
|---|---|---|
| blstm | 0.6077 | 0.6783 |
| tfn | 0.6155 | 0.6334 |
| depdetector | 0.6482 | 0.6508 |
| tamfn | 0.6611 | 0.6984 |
| (STST 2024) | 0.7500 | 0.6623 |
| (DepMamba 2025) | 0.7644 | 0.7320 |
| (CAF-Mamba 2026) | 0.7704 | 0.7487 |

D-Vlog 는 공식 split 이라 직접 비교 가능. **LMVD 는 공식 split 이 없어 참고값일 뿐**이며
반드시 우리가 재구현한 베이스라인끼리만 비교한다.

## 실행
```bash
python -m bench.run --corpus dvlog --model blstm --seeds 5
```
결과는 `results/bench/{corpus}.csv` 에 append 된다.
