# 꽃 모델 v4: ESP 사진 파인튜닝

**상태: 파인튜닝과 평가 완료, 사용할 모델은 미정입니다.** 결과는 [리포트](REPORT.md)를 봅니다. v3 A_base를 사람이 라벨을 그린 ESP 사진으로 파인튜닝하고, 배치가 다른 사진에서 v3와 비교합니다. v3 폴더와 cameras.json은 고치지 않습니다.

| 경로 | 내용 |
|---|---|
| photos/ | ESP 사진 17장의 사본 |
| labels/esp_e1_raw.json | 사람이 그린 상자 원본. 고치지 않습니다 |
| labels/split.csv | 사진별 용도와 md5 |
| labels/masks_body.jsonl | 봉오리 몸통 마스크와 상자. 학습과 IoU 비교에 씁니다 |
| labels/masks.jsonl | 상자 프롬프트만 쓴 첫 시도. 쓰지 않습니다 |
| runs/seg/ft_s0~2 | 파인튜닝 결과 |
| runs/eval | 평가 예측과 결과, 검증에서 고정한 임계값 |

~~~bash
cd scripts
python 10_esp_masks.py     # 첫 시도
python 11_esp_body.py      # 봉오리 몸통 마스크
bash run_train.sh          # 데이터 내보내기와 시드 3개 학습
python 31_esp_eval.py val  # 임계값 고정
python 31_esp_eval.py test # 평가
python 32_phone_val.py     # 휴대폰 val 확인
python 33_esp_figs.py      # 비교 그림
python 12_report.py        # 리포트
~~~
