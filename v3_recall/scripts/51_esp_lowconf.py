# 새 ESP 사진 6장에서 꽃 모델 세 버전의 낮은 신뢰도 후보까지 모아 칸 시트와 사진 그림을 만든다. 입력은 읽기만 한다.
# 사용: python 51_esp_lowconf.py
import glob, importlib.util, sys
from common import *
import paths  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from infer import detect, merge, open_photo, starts, iou_ios

OUT = VER / 'runs/esp_compare/low'; OUT.mkdir(parents=True, exist_ok=True); W = {'v1': V1 / 'runs/det/third_pseudo/weights/best.pt', 'v2': FLOWER / 'v2_seg/runs/seg/first/weights/best.pt', 'v3': VER / 'runs/seg/A_base/weights/best.pt'}; TH = dict(v1=.65, v2=.65, v3=.6); FILES = sorted(glob.glob(str(FLOWER.parent / 'model_git/v10_farm_infer/esp_cam_20261103/inputs/*.jfif')))


def det_box(model, im, tile=1280, overlap=0.25, imgsz=640, batch=32):
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


M = {k: YOLO(str(w)) for k, w in W.items()}; res = {}; font = ImageFont.load_default(); cells = []
for f in FILES:
    im = open_photo(f); name = Path(f).name[:6]; Q = {k: (det_box(m, im) if k == 'v1' else detect(m, im)[0]) for k, m in M.items()}; G = []
    for k, P in Q.items():
        for p in P:
            if p[4] < 0.1:
                continue
            for g in G:
                if iou_ios(p[:4], np.array([g['box']]))[0][0] >= 0.3:
                    g['conf'][k] = max(g['conf'].get(k, 0), round(float(p[4]), 2)); break
            else:
                G.append(dict(box=[round(float(x), 1) for x in p[:4]], conf={k: round(float(p[4]), 2)}))
    G.sort(key=lambda g: -max(g['conf'].values())); res[name] = G; d0 = im.copy(); z = 1600 / im.width; d0 = d0.resize((1600, round(im.height * z))); d = ImageDraw.Draw(d0)
    for i, g in enumerate(G):
        b = g['box']; c = (255, 40, 160) if any(g['conf'].get(k, 0) >= TH[k] for k in TH) else (0, 220, 255); d.rectangle([b[0] * z - 3, b[1] * z - 3, b[2] * z + 3, b[3] * z + 3], outline=c, width=2); d.text((b[0] * z, b[1] * z - 12), str(i + 1), fill=c, font=font); cells.append((name, i + 1, g, im.crop((int(b[0] - 40), int(b[1] - 40), int(b[2] + 40), int(b[3] + 40))).resize((190, 190))))
    d0.save(OUT / f'photo_{name}.jpg', quality=88); print(name, len(G), [(i + 1, [round(x) for x in g['box']], g['conf']) for i, g in enumerate(G)][:14], flush=True)
(OUT / 'low.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
cells = cells[:80]; R_ = (len(cells) + 9) // 10; S = Image.new('RGB', (1900, R_ * 214), (25, 25, 25)); d = ImageDraw.Draw(S)
for j, (n, i, g, t) in enumerate(cells):
    x, y = (j % 10) * 190, (j // 10) * 214; S.paste(t, (x, y + 24)); d.text((x + 2, y + 1), f'{n}#{i}', fill=(255, 255, 255), font=font); d.text((x + 2, y + 12), ' '.join(f'{k}:{v}' for k, v in g['conf'].items()), fill=(255, 220, 90), font=font)
S.save(OUT / 'cells.jpg', quality=86); print('LOW_DONE', len(cells), flush=True)
