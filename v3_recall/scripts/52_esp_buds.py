# 새 ESP 사진 6장에서 육안으로 찾은 봉오리 3개를 사진마다 찾아, 꽃 모델 v1, v2, v3의 신뢰도를 타일 크기별로 잰다. 재학습 없이 축척의 영향만 본다. 입력은 읽기만 한다.
# 봉오리 위치는 0d68e2 사진 기준이고, 다른 사진은 템플릿 맞춤으로 찾는다. 사용: python 52_esp_buds.py
import glob
from common import *
import paths  # noqa: F401
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from infer import detect, merge, open_photo, starts

OUT = VER / 'runs/esp_compare/low'; W = {'v1': V1 / 'runs/det/third_pseudo/weights/best.pt', 'v2': FLOWER / 'v2_seg/runs/seg/first/weights/best.pt', 'v3': VER / 'runs/seg/A_base/weights/best.pt'}; TH = dict(v1=.65, v2=.65, v3=.6); FILES = sorted(glob.glob(str(FLOWER.parent / 'model_git/v10_farm_infer/esp_cam_20261103/inputs/*.jfif')))
REF = next(f for f in FILES if '0d68e2' in f); BUDS = {'A': (1088, 628), 'B': (1093, 1393), 'C': (2045, 510)}; R = 48; ref = np.asarray(open_photo(REF).convert('L'))
print('테스트 라벨 긴 변 중앙값', float(np.median([max(box(r)[2] - box(r)[0], box(r)[3] - box(r)[1]) for r in read(ANN) if {m['photo']: m['split'] for m in read(MANIFEST)}[r['photo']] == 'test'])), flush=True)


def det_box(model, im, tile, overlap=0.25, imgsz=640, batch=32):
    views = []; side = max(im.size); P = []; n = 0
    for y in starts(im.height, tile, overlap):
        for x in starts(im.width, tile, overlap):
            c = Image.new('RGB', (tile, tile), (114, 114, 114)); c.paste(im.crop((x, y, min(x + tile, im.width), min(y + tile, im.height))), (0, 0)); views.append((c.resize((imgsz, imgsz), Image.BILINEAR), x, y, tile / imgsz))
    c = Image.new('RGB', (side, side), (114, 114, 114)); c.paste(im, (0, 0)); views.append((c.resize((imgsz, imgsz), Image.BILINEAR), 0, 0, side / imgsz))
    for i in range(0, len(views), batch):
        vs = views[i:i + batch]
        for v, r in zip(vs, model.predict([t[0] for t in vs], imgsz=imgsz, conf=0.05, iou=0.6, verbose=False)):
            if len(r.boxes):
                t = r.boxes.xyxy.cpu().numpy(); b = t * v[3]; b[:, [0, 2]] += v[1]; b[:, [1, 3]] += v[2]; P.append(np.c_[b, r.boxes.conf.cpu().numpy(), np.zeros(len(b)), np.arange(n, n + len(b))]); n += len(b)
    return merge(np.vstack(P))[:, :5] if P else np.zeros((0, 5))


M = {k: YOLO(str(w)) for k, w in W.items()}; out = {}; tiles = (1280, 960, 640); Sh = []
for f in FILES:
    im = open_photo(f); g = np.asarray(im.convert('L')); nm = Path(f).name[:6]; loc = {}
    for b, (x, y) in BUDS.items():
        T = ref[y - 70:y + 70, x - 70:x + 70]; r = cv2.matchTemplate(g, T, cv2.TM_CCOEFF_NORMED); _, s, _, p = cv2.minMaxLoc(r); loc[b] = (p[0] + 70, p[1] + 70, round(float(s), 2))
    print(nm, '위치', loc, flush=True); row = dict(loc=loc)
    for tl in tiles:
        for k, m in M.items():
            P = det_box(m, im, tl) if k == 'v1' else detect(m, im, tile=tl)[0]; vals = {}; thr = TH[k]
            for b, (x, y, s) in loc.items():
                c = [p[4] for p in P if p[0] - 20 <= x <= p[2] + 20 and p[1] - 20 <= y <= p[3] + 20 and max(p[2] - p[0], p[3] - p[1]) < 220]; vals[b] = round(float(max(c)), 2) if c and s >= 0.45 else (0 if s >= 0.45 else None)
            nthr = int((P[:, 4] >= thr).sum()); near = sum(any(p[0] - 20 <= x <= p[2] + 20 and p[1] - 20 <= y <= p[3] + 20 for b, (x, y, s) in loc.items() if s >= 0.45) for p in P if p[4] >= thr); row[f'{k}_{tl}'] = dict(buds=vals, n=nthr, others=nthr - near)
            print(' ', k, tl, vals, 'n', nthr, 'other', nthr - near, flush=True)
    out[nm] = row
    if nm in ('0d68e2',):
        for b, (x, y, s) in loc.items():
            Sh.append(im.crop((x - 70, y - 70, x + 70, y + 70)).resize((280, 280)))
(OUT / 'buds.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8'); print('BUDS_DONE', flush=True)
