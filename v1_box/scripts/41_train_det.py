# 1단계: 꽃 1클래스 YOLO 탐지 학습. 내보낸 타일(data/det)을 쓴다.
# yolo26s 탐지 사전학습 가중치가 서버에 없어, 구조를 만든 뒤 yolo26s-seg(COCO)에서 맞는 층을 옮긴다.
# 사용: python 41_train_det.py <실행 이름> [--init 가중치] [--epochs 150] [--batch 16]
import argparse, os, sys
from common import *
sys.path.insert(0, str(FLOWER / 'common'))
import paths  # noqa: F401  (opencv 우회가 먼저)
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--init', default=str(paths.WEIGHTS / 'yolo26s-seg.pt'))
ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--data', default=str(VER / 'data' / 'det' / 'flower.yaml')); a = ap.parse_args()
os.chdir(FLOWER / 'common')  # AMP 점검이 ./weights/yolo26n.pt를 찾으므로, 파일이 있는 곳에서 실행해 다시 내려받지 않게 한다
m = YOLO('yolo26s.yaml').load(a.init)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=50, project=str(VER / 'runs' / 'det'), name=a.name,
        exist_ok=True, degrees=15, flipud=0.5, fliplr=0.5, scale=0.5, mosaic=1.0, close_mosaic=10,
        single_cls=True, workers=4, seed=0, plots=True, verbose=False)
print('TRAIN_DONE', flush=True)
