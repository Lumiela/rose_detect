# v1은 찾고 v2만 놓친 test 라벨 중 칸 판정을 받지 않은 것을 번호 칸 시트로 만든다. 파란 박스가 라벨이다. 모델 예측은 그리지 않는다.
# 결과는 runs/seg/first/unjudged/의 sheet.jpg와 errors.csv이고, 번호는 U로 시작한다.
# 사용: python 35_unjudged_sheet.py [--cell 150] [--cols 10]
import argparse, collections as C
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('--cell', type=int, default=150); ap.add_argument('--cols', type=int, default=10); a = ap.parse_args()
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8')); R1 = V1 / 'runs/det/third_pseudo/eval'; R2 = VER / 'runs/seg/first/eval'; OUT = VER / 'runs/seg/first/unjudged'; OUT.mkdir(exist_ok=True)
P1, P2 = J(R1 / 'preds_test.json'), J(R2 / 'preds_test.json'); CONF = J(R1 / 'photo_metrics.json')['val']['chosen']['conf']; GT = C.defaultdict(list)
for r in read(ANN):
    GT[r['photo']].append(r)
JF = {(r['photo'], r['ann_id']) for r in read(V1 / 'labels/review_test_errors_r2.csv') if r['kind'] == 'fn'}


def missed(f, P):
    G = [box(r) for r in GT[f]]; used = set()
    for p in sorted((q for q in P[f] if q[4] >= CONF), key=lambda q: -q[4]):
        v = [ov(p[:4], g)[0] if k not in used else 0 for k, g in enumerate(G)]; k = max(range(len(v)), key=v.__getitem__) if v else -1
        if k >= 0 and v[k] >= 0.5:
            used.add(k)
    return {GT[f][k]['ann_id'] for k in range(len(G)) if k not in used}


rows, cells = [], []
for f in sorted(P1):
    only = missed(f, P2) - missed(f, P1); im = None
    for r in GT[f]:
        if r['ann_id'] in only and r['ann_id'][0] not in 'TV' and (f, r['ann_id']) not in JF:
            im = im or ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); b = box(r); s = max(b[2] - b[0], b[3] - b[1]) * 1.6 + 20; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; x0, y0 = cx - s / 2, cy - s / 2; k = a.cell / s
            c = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((a.cell, a.cell), Image.BILINEAR); d = ImageDraw.Draw(c); d.rectangle([(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k], outline=(0, 120, 255), width=2)
            i = f'U{len(rows) + 1:03d}'; d.rectangle([0, a.cell - 13, a.cell, a.cell], fill=(0, 0, 0)); d.text((2, a.cell - 12), f'{i} {f[9:15]} {r["ann_id"]}', fill=(255, 255, 255)); cells.append(c)
            rows.append(dict(id=i, kind='label', photo=f, ann_id=r['ann_id'], stage=r['stage'], source=r['source'], x0=r['x0'], y0=r['y0'], x1=r['x1'], y1=r['y1']))
write(OUT / 'errors.csv', list(rows[0]), rows); S = Image.new('RGB', (a.cols * a.cell, ((len(cells) + a.cols - 1) // a.cols) * a.cell), (40, 40, 40))
for i, c in enumerate(cells):
    S.paste(c, ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
S.save(OUT / 'sheet.jpg', quality=88); print('cells', len(rows), C.Counter(r['stage'] for r in rows), C.Counter(r['source'] for r in rows), C.Counter(r['photo'][9:15] for r in rows))
