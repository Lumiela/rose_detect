# 마스크 초안을 눈으로 확인하는 번호 시트. 노란 선이 마스크, 파란 선이 프롬프트 박스다.
# 사용: python 11_mask_sheet.py <출력 폴더> [--masks 파일] [--n 100] [--per-stage 25] [--seed 0] [--stage 봉오리,개화중,만개,임시] [--flagged] [--prefix K]
import argparse, collections as C, random
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--masks', default=str(MASKS)); ap.add_argument('--n', type=int, default=100); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--stage', default='')
ap.add_argument('--flagged', action='store_true', help='품질 지표가 나쁜 마스크만'); ap.add_argument('--cell', type=int, default=150); ap.add_argument('--cols', type=int, default=7); ap.add_argument('--prefix', default='K'); ap.add_argument('--per-stage', type=int, default=0, help='단계마다 이 개수만큼 뽑는다'); a = ap.parse_args()
rows = read_masks(a.masks); name = lambda r: r['stage'] or '임시'


def bad(r):
    return r['poly'] is None or r['box_iou'] < 0.7 or r['fill'] < 0.3 or r['outside'] > 0.15


pool = [r for r in rows if (not a.flagged or bad(r)) and (not a.stage or name(r) in a.stage.split(','))]
random.Random(a.seed).shuffle(pool)
if a.per_stage:
    pool = [r for s in STAGES + ['임시'] for r in [q for q in pool if name(q) == s][:a.per_stage]]; a.n = len(pool)
pick = sorted(pool[:a.n], key=lambda r: r['photo']); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cells = []; csv_rows = []; cur = None; im = None
for k, r in enumerate(pick):
    if r['photo'] != cur:
        im = ImageOps.exif_transpose(Image.open(PHOTOS / r['photo'])).convert('RGB'); cur = r['photo']
    b = r['box']; s = max(b[2] - b[0], b[3] - b[1]) * 1.6 + 20; x0, y0 = (b[0] + b[2]) / 2 - s / 2, (b[1] + b[3]) / 2 - s / 2; z = a.cell / s
    c = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((a.cell, a.cell), Image.BILINEAR); d = ImageDraw.Draw(c)
    d.rectangle([(b[0] - x0) * z, (b[1] - y0) * z, (b[2] - x0) * z, (b[3] - y0) * z], outline=(70, 140, 255), width=1)
    if r['poly']:
        d.line([((x - x0) * z, (y - y0) * z) for x, y in r['poly'] + r['poly'][:1]], fill=(255, 235, 0), width=2)
    i = f'{a.prefix}{k + 1:03d}'; d.rectangle([0, a.cell - 13, a.cell, a.cell], fill=(0, 0, 0)); d.text((2, a.cell - 12), f"{i} {r['photo'][9:15]} {r.get('box_iou', 0):.2f}", fill=(255, 255, 255)); cells.append(c)
    csv_rows.append(dict(id=i, photo=r['photo'], key=r['key'], stage=name(r), box_iou=r.get('box_iou'), fill=r.get('fill'), outside=r.get('outside'), score=r.get('score')))
n = (len(cells) + a.cols - 1) // a.cols; s = Image.new('RGB', (a.cols * a.cell, n * a.cell), (40, 40, 40))
for i, c in enumerate(cells):
    s.paste(c, ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
s.save(out / 'sheet_masks.jpg', quality=88); write(out / 'sheet_masks.csv', list(csv_rows[0]), csv_rows); print('pool', len(pool), 'cells', len(cells), s.size, C.Counter(name(r) for r in pick))
