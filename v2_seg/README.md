# 꽃 모델 v2: 세그멘테이션, ID, ID별 단계 분류
**상태: 마스크 초안 생성 완료, 학습 전.** 구조와 비교 기준은 [v1_box 리포트](../v1_box/REPORT.md)를 봅니다.

## 구조 (계획)
1. **세그멘테이션:** YOLO26s-seg, `flower` 1클래스. 봉오리, 개화중, 만개를 모두 합쳐 학습합니다.
2. **ID:** 사진 안에서 꽃마다 번호를 매깁니다.
3. **단계 분류:** ID별로 v1의 분류기를 쓰고, 입력은 마스크를 감싼 박스의 1.25배 크롭입니다.

## 마스크 라벨
- `labels/masks.jsonl`: 박스 프롬프트로 SAM2.1이 만든 초안입니다. **사람이 전부 보지 않았습니다.**
- 박스 라벨 원본은 `../v1_box/labels/` 한 곳을 함께 씁니다.
- 임시 라벨(`source=pseudo`)은 v1의 `third_pseudo`와 같은 출처이며, 판정한 표본 100개는 라벨 또는 제외 영역으로 옮겨 뺐습니다.

## 스크립트
```bash
python scripts/10_make_masks.py                                  # 마스크 생성
python scripts/11_mask_sheet.py runs/mask_check/sample --per-stage 25   # 단계별 표본 시트
python scripts/11_mask_sheet.py runs/mask_check/flagged --flagged --n 70 --prefix X   # 품질 지표가 나쁜 마스크 시트
```
