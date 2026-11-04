# 사용자가 확인하지 않은 train 라벨의 표본으로 확인 화면용 시트를 만든다. 파란 박스가 라벨, 노란 선이 마스크다. 출처는 칸에 쓰지 않는다.
# 결과: labels/user_check_<회차>.csv(번호와 라벨), runs/user_check/<회차>/sheet.jpg. 판정은 화면에서 labels/user_check_<회차>.json으로 저장된다.
# 사용: python 58_user_check_sheet.py r1 [--seed 0]
import argparse, collections as C, json, random
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('round'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--cell', type=int, default=200); ap.add_argument('--cols', type=int, default=10); a = ap.parse_args()
PLAN = [('pseudo', 40, 60, 60), ('pseudo', 60, 80, 80), ('pseudo', 80, 160, 40), ('pseudo', 0, 40, 10), ('pseudo', 160, 1e9, 10), ('legacy', 0, 1e9, 100), ('draft', 0, 1e9, 100)]
masks = {(m['photo'], m['key']): m for m in (json.loads(t) for t in (FLOWER / 'v2_seg/labels/masks.jsonl').read_text(encoding='utf-8').splitlines() if t.strip())}
pool = [r for r in read(VER / 'labels/provenance.csv') if r['split'] == 'train' and r['user_check'] == '없음' and masks.get((r['photo'], r['key']), {}).get('use', True) is not False and (r['photo'], r['key']) in masks]
size = lambda r: max(float(r['x1']) - float(r['x0']), float(r['y1']) - float(r['y0'])); rng = random.Random(a.seed); pick = []
for src, lo, hi, n in PLAN:
    c = [r for r in pool if r['source'] == src and lo <= size(r) < hi]; assert len(c) >= n, (src, lo, len(c)); pick += [dict(r, stratum=f'{src} {int(lo)}-{int(min(hi, 9999))}') for r in rng.sample(c, n)]
rng.shuffle(pick); OUT = VER / 'runs/user_check' / a.round; OUT.mkdir(parents=True, exist_ok=True); rows = []; S = Image.new('RGB', (a.cols * a.cell, ((len(pick) + a.cols - 1) // a.cols) * a.cell), (40, 40, 40)); by = C.defaultdict(list)
for i, r in enumerate(pick):
    by[r['photo']].append((i, r))
for f, items in by.items():
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB')
    for i, r in items:
        b = [float(r[k]) for k in ('x0', 'y0', 'x1', 'y1')]; s = max(b[2] - b[0], b[3] - b[1]) * 1.8 + 24; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; x0, y0 = cx - s / 2, cy - s / 2; k = a.cell / s
        c = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((a.cell, a.cell), Image.BICUBIC); d = ImageDraw.Draw(c); d.rectangle([(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k], outline=(0, 120, 255), width=2)
        q = masks[(f, r['key'])]['poly']
        if q:
            P = [((x - x0) * k, (y - y0) * k) for x, y in q]; d.line(P + P[:1], fill=(255, 230, 0), width=1)
        S.paste(c, ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
for i, r in enumerate(pick):
    rows.append(dict(id=f'C{i + 1:03d}', photo=r['photo'], key=r['key'], stage=r['stage'], source=r['source'], stratum=r['stratum'], x0=r['x0'], y0=r['y0'], x1=r['x1'], y1=r['y1'], size=round(size(r))))
write(VER / 'labels' / f'user_check_{a.round}.csv', list(rows[0]), rows); S.save(OUT / 'sheet.jpg', quality=86); print(len(rows), S.size, C.Counter(r['stratum'] for r in rows), (OUT / 'sheet.jpg').stat().st_size)
