# 탐지 평가의 오류를 눈으로 보는 시트. 빨강 F번호=오탐(라벨 없음), 파랑 M번호=미탐(모델이 놓친 라벨). 번호는 errors_<분할>.csv와 같다.
# 사용: python 43_error_sheet.py <eval 폴더> <분할> [--conf 0.6] [--cell 150] [--cols 10]
import argparse, collections as C, json
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('eval'); ap.add_argument('split'); ap.add_argument('--conf', type=float, default=0.6)
ap.add_argument('--cell', type=int, default=150); ap.add_argument('--cols', type=int, default=10); a = ap.parse_args()
OUT = Path(a.eval); preds = json.loads((OUT / f'preds_{a.split}.json').read_text(encoding='utf-8'))
anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def iou(p, g):
    w = max(0, min(p[2], g[2]) - max(p[0], g[0])); h = max(0, min(p[3], g[3]) - max(p[1], g[1])); i = w * h
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i + 1e-9)


def cell(im, b, color, text):
    w, h = b[2] - b[0], b[3] - b[1]; s = max(w, h) * 1.6 + 20; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; x0, y0 = cx - s / 2, cy - s / 2
    c = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((a.cell, a.cell), Image.BILINEAR); d = ImageDraw.Draw(c); k = a.cell / s
    d.rectangle([(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k], outline=color, width=2)
    d.rectangle([0, a.cell - 13, a.cell, a.cell], fill=(0, 0, 0)); d.text((2, a.cell - 12), text, fill=(255, 255, 255))
    return c


fp, fn, rows = [], [], []
for f, P in preds.items():
    P = sorted([p for p in P if p[4] >= a.conf], key=lambda p: -p[4]); gt = anns[f]; used = set(); hms = f[9:15]
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB')
    for p in P:
        v = [(iou(p, box(g)), j) for j, g in enumerate(gt) if j not in used]; best = max(v) if v else (0, -1)
        if best[0] >= 0.5:
            used.add(best[1]); continue
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in igns[f]):
            near = max((iou(p, box(g)) for g in gt), default=0)
            rows.append(dict(id=f'F{len(fp) + 1:03d}', kind='fp', photo=f, ann_id='', stage='', x0=round(p[0]), y0=round(p[1]), x1=round(p[2]), y1=round(p[3]), conf=round(p[4], 3), iou=round(near, 2)))
            fp.append(cell(im, p, (255, 0, 0), f'{rows[-1]["id"]} {hms} {p[4]:.2f} i{near:.1f}'))
    for j, g in enumerate(gt):
        if j not in used:
            b = box(g); rows.append(dict(id=f'M{len(fn) + 1:03d}', kind='fn', photo=f, ann_id=g['ann_id'], stage=g['stage'], x0=g['x0'], y0=g['y0'], x1=g['x1'], y1=g['y1'], conf='', iou=''))
            fn.append(cell(im, b, (0, 120, 255), f"{rows[-1]['id']} {hms} {g['ann_id']} {g['stage'][0]} {int(max(b[2]-b[0], b[3]-b[1]))}"))
write(OUT / f'errors_{a.split}.csv', list(rows[0]), rows)
for name, cells in (('fp', fp), ('fn', fn)):
    h = (len(cells) + a.cols - 1) // a.cols; s = Image.new('RGB', (a.cols * a.cell, max(h, 1) * a.cell), (40, 40, 40))
    for i, c in enumerate(cells):
        s.paste(c, ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
    s.save(OUT / f'sheet_{a.split}_{name}.jpg', quality=88); print(name, len(cells), s.size)
