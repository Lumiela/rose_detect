# ESP 학습 데이터를 만든다. 타일 880, 겹침 0.25, 입력 640에 사진 전체 보기 1장을 더한다. 라벨은 masks_body.jsonl의 봉오리 몸통 다각형이다. 평가와 제외 사진은 넣지 않는다.
# 사용: python 20_esp_export.py
import collections as C, shutil
from common import *
from PIL import Image
OUT = VER / 'data'; TILE, OV, SZ = 880, .25, 640
if OUT.exists():
    shutil.rmtree(OUT)
rows = C.defaultdict(list)
for t in (VER / 'labels' / 'masks_body.jsonl').read_text(encoding='utf-8').splitlines():
    r = json.loads(t); rows[r['photo']].append(r)
SP = splits(); st = C.Counter()


def starts(size, tile):
    return [0] if size <= tile else list(range(0, size - tile, int(tile * (1 - OV)))) + [size - tile]


def area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2


def clip(poly, x0, y0, x1, y1):
    for inside, cross in [(lambda p: p[0] >= x0, lambda p, q: (x0, p[1] + (q[1] - p[1]) * (x0 - p[0]) / (q[0] - p[0]))), (lambda p: p[0] <= x1, lambda p, q: (x1, p[1] + (q[1] - p[1]) * (x1 - p[0]) / (q[0] - p[0]))),
                          (lambda p: p[1] >= y0, lambda p, q: (p[0] + (q[0] - p[0]) * (y0 - p[1]) / (q[1] - p[1]), y0)), (lambda p: p[1] <= y1, lambda p, q: (p[0] + (q[0] - p[0]) * (y1 - p[1]) / (q[1] - p[1]), y1))]:
        out = []
        for i, p in enumerate(poly):
            q = poly[i - 1]
            if inside(p):
                if not inside(q):
                    out.append(cross(q, p))
                out.append(p)
            elif inside(q):
                out.append(cross(q, p))
        poly = out
        if not poly:
            return []
    return poly


for split in ('train', 'val'):
    (OUT / 'images' / split).mkdir(parents=True); (OUT / 'labels' / split).mkdir(parents=True)
for p, sp in SP.items():
    if sp not in ('train', 'val'):
        continue
    im = Image.open(PHOTOS / p).convert('RGB'); views = [(x, y, TILE) for y in starts(im.height, TILE) for x in starts(im.width, TILE)] + [(0, 0, max(im.size))]
    for i, (x, y, side) in enumerate(views):
        c = Image.new('RGB', (side, side), (114, 114, 114)); c.paste(im.crop((x, y, min(x + side, im.width), min(y + side, im.height))), (0, 0)); lines = []
        for r in rows[p]:
            q = clip([tuple(v) for v in r['poly']], x, y, x + side, y + side)
            if len(q) < 3 or area(q) / max(area(r['poly']), 1) < .5:
                if q:
                    st['cut_' + sp] += 1
                continue
            lines.append('0 ' + ' '.join(f'{(u - x) / side:.5f} {(v - y) / side:.5f}' for u, v in q))
        stem = f'{p[:6]}_{i:02d}'; c.resize((SZ, SZ), Image.BILINEAR).save(OUT / 'images' / sp / f'{stem}.jpg', quality=92); (OUT / 'labels' / sp / f'{stem}.txt').write_text('\n'.join(lines), encoding='ascii'); st['tiles_' + sp] += 1; st['objects_' + sp] += len(lines); st['empty_' + sp] += not lines
(OUT / 'flower.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: flower\n', encoding='utf-8'); print('EXPORT_DONE', dict(st), flush=True)
