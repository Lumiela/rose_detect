# 시험 학습에 넣은 임시 라벨(사람이 보지 않은 예측)의 표본 시트. 무작위로 뽑아 사람이 판정하면 임시 라벨의 정밀도를 추정할 수 있다.
# 사용: python 49_pseudo_sample.py <preds_train.json> <출력 폴더> [--n 100] [--hi 0.7] [--seed 0]
import argparse, json, random
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('preds'); ap.add_argument('out'); ap.add_argument('--n', type=int, default=100); ap.add_argument('--hi', type=float, default=0.7); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--new-since', help='이 예측의 hi 이상 후보와 겹치지 않는 새 후보만 뽑는다')
ap.add_argument('--cell', type=int, default=150); ap.add_argument('--cols', type=int, default=7); a = ap.parse_args()
P = json.loads(Path(a.preds).read_text(encoding='utf-8')); man = {m['photo']: m for m in read(MANIFEST)}; rows = read(ANN) + read(IGN) + read(REJ); pool = []
OLD = json.loads(Path(a.new_since).read_text(encoding='utf-8')) if a.new_since else {}


def ov(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    x, y = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (x + y - i + 1e-9), i / min(x, y)


for f, ps in P.items():
    if man[f]['label_status'] != 'partial':
        continue
    gs = [box(r) for r in rows if r['photo'] == f]
    gs += [q[:4] for q in OLD.get(f, []) if q[4] >= a.hi]
    for p in ps:
        if p[4] >= a.hi and not any(i >= 0.3 or s >= 0.6 for i, s in (ov(p[:4], g) for g in gs)):
            pool.append((f, p))
random.Random(a.seed).shuffle(pool); pick = sorted(enumerate(pool[:a.n]), key=lambda t: t[1][0])
order = sorted(pool[:a.n], key=lambda t: t[0]); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cells = []; csv_rows = []; cur = None; im = None
for k, (f, p) in enumerate(order):
    if f != cur:
        im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); cur = f
    w, h = p[2] - p[0], p[3] - p[1]; s = max(w, h) * 1.6 + 20; x0, y0 = (p[0] + p[2]) / 2 - s / 2, (p[1] + p[3]) / 2 - s / 2; z = a.cell / s
    c = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((a.cell, a.cell), Image.BILINEAR); d = ImageDraw.Draw(c)
    d.rectangle([(p[0] - x0) * z, (p[1] - y0) * z, (p[2] - x0) * z, (p[3] - y0) * z], outline=(255, 0, 0), width=2)
    d.rectangle([0, a.cell - 13, a.cell, a.cell], fill=(0, 0, 0)); d.text((2, a.cell - 12), f'P{k + 1:03d} {f[9:15]} {p[4]:.2f}', fill=(255, 255, 255)); cells.append(c)
    csv_rows.append(dict(id=f'P{k + 1:03d}', photo=f, x0=round(p[0]), y0=round(p[1]), x1=round(p[2]), y1=round(p[3]), conf=round(p[4], 3)))
n = (len(cells) + a.cols - 1) // a.cols; s = Image.new('RGB', (a.cols * a.cell, n * a.cell), (40, 40, 40))
for i, c in enumerate(cells):
    s.paste(c, ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
s.save(out / 'sheet_pseudo_sample.jpg', quality=88); write(out / 'pseudo_sample.csv', list(csv_rows[0]), csv_rows); print('pool', len(pool), 'cells', len(cells), s.size)
