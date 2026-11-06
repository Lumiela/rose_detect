# v3 A_base를 ESP 사진으로 파인튜닝한다. 설정은 평가 전에 고정했다. 백본 고정, 낮은 학습률, 증강은 v3와 같다. 사용: python 30_esp_train.py <이름> <시드>
import os, sys
from common import *
import paths  # noqa: F401
from ultralytics import YOLO
name, seed = sys.argv[1], int(sys.argv[2]); os.chdir(FLOWER / 'common')
m = YOLO(str(V3 / 'runs/seg/A_base/weights/best.pt'))
m.train(data=str(VER / 'data/flower.yaml'), epochs=60, imgsz=640, batch=16, patience=60, project=str(VER / 'runs/seg'), name=name, exist_ok=True, optimizer='SGD', lr0=.002, lrf=.1, momentum=.937, warmup_epochs=1, freeze=10,
        degrees=15, flipud=.5, fliplr=.5, scale=.5, mosaic=1., close_mosaic=10, single_cls=True, workers=4, seed=seed, plots=False, verbose=False)
print('TRAIN_DONE', name, flush=True)
