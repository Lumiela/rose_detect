# 1단계: 꽃 1클래스 세그멘테이션 학습. v1 탐지 학습과 같은 증강과 회차를 쓴다.
# 사용: python 30_train_seg.py <실행 이름> [--epochs 80] [--batch 16] [--data 파일]
import argparse, os
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--init', default=str(paths.WEIGHTS / 'yolo26s-seg.pt')); ap.add_argument('--epochs', type=int, default=80); ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--data', default=str(VER / 'data' / 'flower.yaml')); ap.add_argument('--mask-ratio', type=int, default=4); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
os.chdir(FLOWER / 'common')  # AMP 점검이 ./weights/yolo26n.pt를 찾으므로, 파일이 있는 곳에서 실행해 다시 내려받지 않게 한다
m = YOLO('yolo26s-seg.yaml').load(a.init)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=50, project=str(VER / 'runs' / 'seg'), name=a.name,
        exist_ok=True, degrees=15, flipud=0.5, fliplr=0.5, scale=0.5, mosaic=1.0, close_mosaic=10,
        single_cls=True, workers=4, seed=a.seed, mask_ratio=a.mask_ratio, plots=True, verbose=False)
print('TRAIN_DONE', flush=True)
