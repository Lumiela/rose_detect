# 파인튜닝이 휴대폰 사진 성능을 떨어뜨리는지 휴대폰 val 타일로 확인한다. test는 열지 않는다. 사용: python 32_phone_val.py
import os
from common import *
import paths  # noqa: F401
from ultralytics import YOLO
os.chdir(FLOWER / 'common'); out = {}
for n, w in [('v3', V3 / 'runs/seg/A_base/weights/best.pt')] + [(f'ft_s{s}', VER / f'runs/seg/ft_s{s}/weights/best.pt') for s in (0, 1, 2)]:
    r = YOLO(str(w)).val(data=str(V3 / 'data/flower.yaml'), split='val', imgsz=640, batch=16, single_cls=True, plots=False, verbose=False, project=str(VER / 'runs/phone_val'), name=n, exist_ok=True)
    out[n] = dict(box_map50=round(float(r.box.map50), 3), box_p=round(float(r.box.mp), 3), box_r=round(float(r.box.mr), 3), mask_map50=round(float(r.seg.map50), 3)); print(n, out[n], flush=True)
(VER / 'runs/eval/phone_val.json').write_text(json.dumps(out), encoding='utf-8'); print('PHONE_DONE', flush=True)
