#!/bin/bash
# v3 세 조건을 차례로 학습하고 val로만 평가한다. test는 조건을 정한 뒤 한 번만 본다.
cd "$(dirname "$0")"; L=../runs/logs; mkdir -p $L
echo START A $(date) >> $L/all.log
python 30_train_seg.py A_base --epochs 80 > $L/A_base.log 2>&1 && python 31_eval_seg.py ../runs/seg/A_base/weights/best.pt ../runs/seg/A_base/eval_val --splits val > $L/A_base_eval.log 2>&1
echo DONE A $? $(date) >> $L/all.log
python 30_train_seg.py C_mask2 --epochs 80 --mask-ratio 2 > $L/C_mask2.log 2>&1 && python 31_eval_seg.py ../runs/seg/C_mask2/weights/best.pt ../runs/seg/C_mask2/eval_val --splits val > $L/C_mask2_eval.log 2>&1
echo DONE C $? $(date) >> $L/all.log
python 20_export_seg.py --out ../data_960 --imgsz 960 --splits train,val > $L/export_960.log 2>&1 && python 30_train_seg.py B_960 --epochs 80 --imgsz 960 --data "$(cd ..; pwd)/data_960/flower.yaml" > $L/B_960.log 2>&1 && python 31_eval_seg.py ../runs/seg/B_960/weights/best.pt ../runs/seg/B_960/eval_val --splits val --imgsz 960 > $L/B_960_eval.log 2>&1
echo DONE B $? $(date) >> $L/all.log
echo ALLDONE >> $L/all.log
