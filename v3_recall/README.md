# 꽃 모델 v3: 팀 교차 판정 라벨로 재현율 개선

**상태: A_base 학습과 test 평가 완료, 최종 모델 결정 전입니다.** 결과는 [리포트](REPORT.md)를 봅니다. v2와 같은 구조(세그멘테이션, ID, 단계 분류)에서 학습 라벨을 고쳐 작은 봉오리 재현율을 개선합니다.

## 라벨
v1_box/labels를 복사해 사람 판정을 반영한 사본입니다. v1_box/labels는 고치지 않습니다. val, test 라벨은 v1과 같아야 하고 평가 스크립트가 확인합니다. val, test를 고칠 때는 v1, v2도 새 라벨로 다시 평가합니다.

| 파일 | 내용 |
|---|---|
| labels/annotations.csv | 라벨 원본. 새 라벨은 ann_id H, source team |
| labels/ignore_regions.csv, pseudo_rejected.csv | 제외 영역, 차단 목록 |
| labels/team_added.csv | 새 라벨별 조원, Claude, 사용자 판정 |
| labels/team_skipped.csv, user_check_applied.csv | 겹쳐서 뺀 칸, 표본 확인 13칸 처리 |
| labels/masks.jsonl | SAM2.1 마스크. v2 마스크는 박스가 같으면 다시 씀 |
| labels/provenance.csv | 라벨별 출처와 사람 확인 범위 |

판정 원본은 ../human_label/에 있습니다.

## 실행
~~~bash
cd scripts
python 01_apply_team_check.py [--apply]   # 라벨 사본 만들기, 기본은 미리 보기
python 10_make_masks.py                   # v2에 없는 칸만 마스크 생성
python 10_make_masks.py --retry
python 02_provenance.py
python 20_export_seg.py
python 30_train_seg.py <이름> [--imgsz 960] [--mask-ratio 2] [--seed 0]
python 31_eval_seg.py <가중치> <출력 폴더>
~~~
