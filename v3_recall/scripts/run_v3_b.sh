#!/bin/bash
# 입력 960 조건만 다시 돌린다. 30_train_seg.py가 작업 폴더를 옮기므로 데이터 경로는 절대 경로로 준다.
cd "$(dirname "$0")"; L=../runs/logs; D=$(cd ..; pwd)/data_960/flower.yaml
echo START B2 $(date) >> $L/all.log
python 30_train_seg.py B_960 --epochs 80 --imgsz 960 --data "$D" > $L/B_960.log 2>&1 && python 31_eval_seg.py ../runs/seg/B_960/weights/best.pt ../runs/seg/B_960/eval_val --splits val --imgsz 960 > $L/B_960_eval.log 2>&1
echo DONE B2 $? $(date) >> $L/all.log
