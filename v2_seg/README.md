# 꽃 모델 v2: 세그멘테이션, ID, ID별 단계 분류
**상태: 세그멘테이션 학습, v1 비교, ID 부여 완료. 최종 모델은 아직 정하지 않았습니다.** 결과는 [리포트](REPORT.md)를 봅니다.

## 구조
1. **세그멘테이션:** YOLO26s-seg, `flower` 1클래스. 봉오리, 개화중, 만개를 모두 합쳐 학습합니다.
2. **ID:** 사진 안에서 꽃마다 번호를 매깁니다. 위에서 아래, 왼쪽에서 오른쪽 순서이고 형식은 `<사진>_F0001`입니다.
3. **단계 분류:** ID별로 v1의 분류기를 쓰고, 입력은 모델 박스의 1.25배 크롭입니다. 마스크를 감싼 박스는 평가 점수가 더 낮아 쓰지 않습니다.

## ID 부여
```bash
python scripts/40_flower_ids.py <사진 또는 폴더> <출력 폴더> --sheet    # 임의의 사진
python scripts/40_flower_ids.py --splits val,test runs/ids/first --sheet   # 라벨이 완성된 분할
python scripts/41_check_ids.py runs/ids/first/flowers.json runs/seg/first/eval   # 평가 결과와 일치, 합쳐짐과 쪼개짐 확인 (check.json)
```
| 출력 | 내용 |
|---|---|
| `<사진>_ids.jpg` | 마스크와 번호. 초록은 봉오리, 노랑은 개화중, 분홍은 만개입니다. |
| `<사진>_sheet.jpg` | `--sheet`일 때 ID별 크롭, 단계, 단계 확률 |
| `flowers.csv` | 꽃마다 ID, 박스, 중심, 탐지 신뢰도, 단계, 단계 확률, 마스크 넓이, 볼록도, 가장자리 접촉 |
| `flowers.json` | `flowers.csv` 내용과 마스크 다각형 |
| `summary.csv` | 사진별 단계 개수와 처리 시간 |

- **ID는 사진 한 장 안에서만 유효합니다.** 다른 사진의 같은 꽃과 이어지지 않으므로 시간에 따른 추적은 아직 할 수 없습니다.
- **단계 확률:** 봉오리는 맞은 경우와 틀린 경우의 중앙값이 0.526과 0.515로 거의 같습니다. 봉오리 예측의 신뢰도로 쓰지 않습니다.
- **볼록도:** 0.8 미만 14개를 봤더니 대부분 잎에 가려진 한 송이였습니다. 합쳐진 꽃을 찾는 기준으로 쓰지 않습니다.
- `scripts/infer.py`는 `31_eval_seg.py`와 `40_flower_ids.py`가 함께 쓰는 추론 코드입니다.

## 마스크 라벨
- `labels/masks.jsonl`: 박스 프롬프트로 SAM2.1이 만든 초안입니다. **사람이 전부 보지 않았습니다.**
- 박스 라벨 원본은 `../v1_box/labels/` 한 곳을 함께 씁니다.
- 2026-11-04에 v1 라벨 원본에서 지운 박스의 마스크 16개를 뺐습니다 (`scripts/13_drop_masks.py`). 2차 판정으로 추가한 test 라벨 43개는 마스크가 없습니다. 모델은 정리 전 데이터로 학습했습니다.
- 임시 라벨(`source=pseudo`)은 v1의 `third_pseudo`와 같은 출처이며, 판정한 표본 100개는 라벨 또는 제외 영역으로 옮겨 뺐습니다.

## 스크립트
```bash
python scripts/10_make_masks.py                                  # 마스크 생성
python scripts/11_mask_sheet.py runs/mask_check/sample --per-stage 25   # 단계별 표본 시트
python scripts/11_mask_sheet.py runs/mask_check/flagged --flagged --n 70 --prefix X   # 품질 지표가 나쁜 마스크 시트
python scripts/10_make_masks.py --retry                          # 실패 마스크 재생성
python scripts/20_export_seg.py                                  # 학습 데이터 내보내기
python scripts/30_train_seg.py first --epochs 80                 # 학습
python scripts/31_eval_seg.py runs/seg/first/weights/best.pt runs/seg/first/eval   # 사진 단위 평가
python scripts/32_compare_v1.py runs/seg/first/eval runs/seg/first/compare         # v1 비교
python scripts/12_report.py                                      # 리포트
```
