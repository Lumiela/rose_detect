# v1 탐지 모델과 v2 세그멘테이션 모델의 추론 시간과 크기를 같은 조건으로 잰다. 사진 열기, 화면 준비, 결과 합치기는 재지 않는다. 첫 바퀴는 예열로 버린다.
# 결과는 runs/seg/first/compare/timing.json에 남긴다. 사용: python 34_time_models.py
import time
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
from PIL import Image
from ultralytics import YOLO
from infer import open_photo, starts

ph = [PHOTOS / m['photo'] for m in read(MANIFEST) if m['split'] in ('val', 'test') and m['label_status'] in ('draft', 'verified')]; res = {}
for name, w in (('v1', V1 / 'runs/det/third_pseudo/weights/best.pt'), ('v2', VER / 'runs/seg/first/weights/best.pt')):
    m = YOLO(str(w)); n = sum(p.numel() for p in m.model.parameters()); T = []
    for rnd in range(2):
        for f in ph:
            im = open_photo(f); v = []; s = max(im.size)
            for y in starts(im.height, 1280, 0.25):
                for x in starts(im.width, 1280, 0.25):
                    c = Image.new('RGB', (1280, 1280), (114, 114, 114)); c.paste(im.crop((x, y, min(x + 1280, im.width), min(y + 1280, im.height))), (0, 0)); v.append(c.resize((640, 640), Image.BILINEAR))
            c = Image.new('RGB', (s, s), (114, 114, 114)); c.paste(im, (0, 0)); v.append(c.resize((640, 640), Image.BILINEAR)); t = time.time()
            for i in range(0, len(v), 32):
                m.predict(v[i:i + 32], imgsz=640, conf=0.05, iou=0.6, verbose=False, **(dict(retina_masks=True) if name == 'v2' else {}))
            T.append(time.time() - t)
    T = T[len(ph):]; m.fuse(); res[name] = dict(params_all=n, params_fused=sum(p.numel() for p in m.model.parameters()), size_mb=round(Path(w).stat().st_size / 1e6, 1), views=len(v), median_s=round(float(np.median(T)), 3), mean_s=round(float(np.mean(T)), 3)); print(name, res[name], flush=True)
out = VER / 'runs/seg/first/compare/timing.json'; out.write_text(json.dumps(res, indent=1), encoding='utf-8'); print('TIMING_DONE', flush=True)
