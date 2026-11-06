# 꽃 찾기, ID 부여, ID별 단계 분류. 꽃마다 번호, 위치, 마스크, 단계를 남긴다. 파일 형식은 잎 모델 v7_leafid와 같은 방식이다.
# ID는 사진 안에서만 쓰는 번호다. 위에서 아래, 왼쪽에서 오른쪽 순서이며 다른 사진의 같은 꽃과 이어지지 않는다.
# 사용: python 40_flower_ids.py <사진 또는 폴더> <출력 폴더> [--camera esp] [--seg 가중치] [--cls 가중치] [--conf 값] [--sheet]
#       python 40_flower_ids.py --splits val,test <출력 폴더> [--sheet]   (라벨이 완성된 분할의 사진)
# 출력: <사진>_ids.jpg, <사진>_sheet.jpg(--sheet), flowers.csv, flowers.json(다각형 포함), summary.csv
import argparse, collections as C, time
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from infer import classify, crop_box, detect, load_cls, open_photo

EXTS = {'.jpg', '.jpeg', '.jfif', '.png'}; NAMES = ['bud', 'opening', 'full']; COLORS = [(60, 220, 90), (255, 200, 0), (255, 60, 170)]
# 기본 가중치는 저장소 weights/의 사본이다 (v3 runs/seg/A_base, v1 runs/cls/effnetv2s_r2와 md5 같음)
ap = argparse.ArgumentParser(); ap.add_argument('paths', nargs='+', help='[사진 또는 폴더] 출력 폴더'); ap.add_argument('--splits'); ap.add_argument('--camera', default='phone', help='cameras.json의 설정'); ap.add_argument('--conf', type=float, default=None, help='기본은 카메라 설정 값')
ap.add_argument('--seg', default=str(FLOWER / 'weights/flower_seg_v3.pt')); ap.add_argument('--cls', default=str(FLOWER / 'weights/flower_cls_v1r2.pt')); ap.add_argument('--sheet', action='store_true'); a = ap.parse_args(); CAM = camera(a.camera); a.conf = CAM['conf'] if a.conf is None else a.conf; print('camera', a.camera, CAM, flush=True)
OUT = Path(a.paths[-1]); OUT.mkdir(parents=True, exist_ok=True)
if a.splits:
    photos = [PHOTOS / m['photo'] for m in read(MANIFEST) if m['split'] in a.splits.split(',') and m['label_status'] in ('draft', 'verified')]
else:
    src = Path(a.paths[0]); photos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
seg, net = YOLO(a.seg), load_cls(a.cls)


def mask_stats(q):
    # 다각형을 칠해 (넓이, 볼록도)를 잰다. 볼록도 0.8 미만은 대부분 잎에 가려진 한 송이라 합쳐짐 판단에 쓰지 않는다
    if len(q) < 3:
        return 0, 0.0
    o = np.floor(q.min(0)).astype(int); w, h = np.ceil(q.max(0)).astype(int) - o + 1; m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(q - o).astype(np.int32)], 1)
    area = int(m.sum()); pts = cv2.findNonZero(m)
    return area, round(min(area / max(cv2.contourArea(cv2.convexHull(pts)), 1.0), 1.0), 3) if pts is not None else 0.0


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_ids(im, rows, polys):
    long = max(im.size); layer = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer); F = font(long // 90)
    for r, q in sorted(zip(rows, polys), key=lambda t: -t[0]['area']):  # 작은 꽃이 위로 오게
        c = COLORS[r['stage_idx']]; xy = [tuple(p) for p in q.tolist()]
        if len(xy) >= 3:
            d.polygon(xy, fill=c + (70,)); d.line(xy + xy[:1], fill=c + (255,), width=max(2, long // 1000))
    for r in rows:  # 번호보다 작은 꽃은 가리지 않게 박스 위에 쓴다
        y = r['cy'] if max(r['x1'] - r['x0'], r['y1'] - r['y0']) >= long // 30 else r['y0'] - long // 150
        d.text((r['cx'], y), str(r['idx']), font=F, fill=(255, 255, 255, 255), anchor='mm' if y == r['cy'] else 'mb', stroke_width=max(2, long // 800), stroke_fill=(0, 0, 0, 255))
    out = Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB'); d = ImageDraw.Draw(out); F = font(long // 45); n = C.Counter(r['stage_idx'] for r in rows); x = 12; h = long // 45 + 24; d.rectangle((0, 0, im.width, h), fill=(20, 20, 20))
    for i in range(3):
        t = f'{NAMES[i]} {n[i]}'; d.text((x, 10), t, font=F, fill=COLORS[i]); x += int(d.textlength(t, font=F)) + long // 40
    d.text((x, 10), f'total {len(rows)}', font=F, fill=(255, 255, 255)); s = 2400 / long
    return out.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else out


def draw_sheet(im, rows, polys, cols=10, cell=190):
    H = 24; S = Image.new('RGB', (cols * cell, ((len(rows) + cols - 1) // cols) * (cell + H) or 1), (30, 30, 30)); d = ImageDraw.Draw(S); F = font(16)
    for k, (r, q) in enumerate(zip(rows, polys)):
        b = crop_box([r['x0'], r['y0'], r['x1'], r['y1']]); t = im.crop(b); s = cell / t.width; t = t.resize((cell, cell), Image.BILINEAR); e = ImageDraw.Draw(t); c = COLORS[r['stage_idx']]
        if len(q) >= 3:
            e.line([tuple(p) for p in ((q - b[:2]) * s).tolist()] + [tuple(((q[0] - b[:2]) * s).tolist())], fill=c, width=2)
        x, y = (k % cols) * cell, (k // cols) * (cell + H); S.paste(t, (x, y + H)); d.text((x + 4, y + 3), f"{r['idx']} {NAMES[r['stage_idx']]} {r['stage_prob']:.2f}", font=F, fill=c)
    return S


rows_all, js, summary = [], [], []
for f in photos:
    im = open_photo(f); t0 = time.time(); P, Q = detect(seg, im, tile=CAM['tile'], overlap=CAM['overlap'], imgsz=CAM['imgsz']); t1 = time.time(); k = P[:, 4] >= a.conf; P = P[k]; Q = [q for q, y in zip(Q, k) if y]; pr = classify(net, im, P); t2 = time.time()
    row = max(im.size) // 20; order = sorted(range(len(P)), key=lambda i: (int((P[i, 1] + P[i, 3]) / 2) // row, (P[i, 0] + P[i, 2]) / 2)); rows = []; polys = []
    for idx, i in enumerate(order, 1):
        p = P[i]; y = int(pr[i].argmax()); area, sol = mask_stats(Q[i])
        rows.append(dict(id=f'{f.stem}_F{idx:04d}', photo=f.name, idx=idx, x0=round(float(p[0]), 3), y0=round(float(p[1]), 3), x1=round(float(p[2]), 3), y1=round(float(p[3]), 3), cx=round(float(p[0] + p[2]) / 2, 1), cy=round(float(p[1] + p[3]) / 2, 1),
                         conf=round(float(p[4]), 3), stage=STAGES[y], stage_idx=y, stage_prob=round(float(pr[i, y]), 3), prob_bud=round(float(pr[i, 0]), 3), prob_opening=round(float(pr[i, 1]), 3), prob_full=round(float(pr[i, 2]), 3),
                         area=area, solidity=sol, touches_border=bool(p[0] <= 2 or p[1] <= 2 or p[2] >= im.width - 2 or p[3] >= im.height - 2)))
        polys.append(np.round(Q[i]).astype(int))
    rows_all += rows; js.append(dict(photo=f.name, width=im.width, height=im.height, flowers=[dict(r, poly=q.tolist()) for r, q in zip(rows, polys)]))
    draw_ids(im, rows, polys).save(OUT / f'{f.stem}_ids.jpg', quality=88)
    if a.sheet and rows:
        draw_sheet(im, rows, polys).save(OUT / f'{f.stem}_sheet.jpg', quality=88)
    n = C.Counter(r['stage'] for r in rows); summary.append(dict(photo=f.name, width=im.width, height=im.height, total=len(rows), **{s: n[s] for s in STAGES}, detect_s=round(t1 - t0, 2), classify_s=round(t2 - t1, 2)))
    print(f.name, summary[-1], flush=True)
if rows_all:
    write(OUT / 'flowers.csv', list(rows_all[0]), rows_all)
write(OUT / 'summary.csv', list(summary[0]), summary); (OUT / 'flowers.json').write_text(json.dumps(js, ensure_ascii=False), encoding='utf-8'); print('IDS_DONE', len(photos), len(rows_all), flush=True)
