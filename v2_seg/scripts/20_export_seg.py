# 세그멘테이션 학습 데이터를 만든다. 타일과 축척, 잘린 꽃 처리는 v1 탐지 내보내기와 같고, 라벨만 박스 대신 다각형이다.
# 클래스는 flower 하나다. 마스크가 실패한 꽃(use=false, 다각형 없음)과 제외 영역은 회색으로 가려 배경으로 학습되지 않게 한다.
# 일부 라벨 사진은 v1 third_pseudo와 같이 임시 라벨을 쓰고, 신뢰도 lo~hi 예측은 회색으로 가린다.
# 사용: python 20_export_seg.py [--out data] [--splits train,val,test] [--lo 0.3] [--hi 0.7] [--limit 사진 수]
import argparse, collections as C, shutil
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(VER / 'data')); ap.add_argument('--splits', default='train,val,test'); ap.add_argument('--lo', type=float, default=0.3); ap.add_argument('--hi', type=float, default=0.7); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--tile', type=int, default=1280); ap.add_argument('--overlap', type=float, default=0.25); ap.add_argument('--imgsz', type=int, default=640); a = ap.parse_args()
OUT = Path(a.out).resolve(); GRAY = (114, 114, 114)
if OUT.exists():
    shutil.rmtree(OUT)
rows = C.defaultdict(list); igns = C.defaultdict(list); P = json.loads(PSEUDO_PREDS.read_text(encoding='utf-8'))
for r in read_masks():
    rows[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def starts(size):
    if size <= a.tile:
        return [0]
    return list(range(0, size - a.tile, int(a.tile * (1 - a.overlap)))) + [size - a.tile]


def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2


def clip(poly, x0, y0, x1, y1):
    # 다각형을 사각형으로 자른다 (Sutherland-Hodgman)
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


def tile(im, x0, y0, side, objs, grays, stem, split, st):
    crop = Image.new('RGB', (side, side), GRAY); crop.paste(im.crop((x0, y0, min(x0 + side, im.width), min(y0 + side, im.height))), (0, 0))
    d = ImageDraw.Draw(crop); lines = []

    def fill(b):
        c = (max(b[0], x0), max(b[1], y0), min(b[2], x0 + side), min(b[3], y0 + side))
        if c[0] < c[2] and c[1] < c[3]:
            d.rectangle([c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0], fill=GRAY)

    for b in grays:
        fill(b)
    for b, poly in objs:
        c = clip([tuple(p) for p in poly], x0, y0, x0 + side, y0 + side)
        if not c:
            continue
        if len(c) < 3 or area(c) / max(area(poly), 1) < 0.5:
            fill(b); continue
        lines.append('0 ' + ' '.join(f'{(px - x0) / side:.5f} {(py - y0) / side:.5f}' for px, py in c))
    crop.resize((a.imgsz, a.imgsz), Image.BILINEAR).save(OUT / 'images' / split / f'{stem}.jpg', quality=92)
    (OUT / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines), encoding='ascii'); st['tiles_' + split] += 1; st['objects_' + split] += len(lines)


st = C.Counter(); man = [m for m in read(MANIFEST) if m['split'] in a.splits.split(',') and m['label_status'] != 'excluded']
for s in a.splits.split(','):
    (OUT / 'images' / s).mkdir(parents=True); (OUT / 'labels' / s).mkdir(parents=True)
if a.limit:
    man = man[:a.limit]
for m in man:
    f = m['photo']; objs, grays = [], list(igns[f]); boxes = [r['box'] for r in rows[f]]
    if m['split'] != 'train' and m['label_status'] == 'partial':
        continue
    for r in rows[f]:
        if r['poly'] and r.get('use', True):
            objs.append((r['box'], r['poly']))
        else:
            grays.append(r['box']); st['gray_failed'] += 1
    if m['label_status'] == 'partial':
        for p in sorted(P.get(f, []), key=lambda q: -q[4]):
            if a.lo <= p[4] < a.hi:
                c = [ov(p[:4], g) for g in boxes]
                if not any(i >= 0.3 or s >= 0.3 for i, s in c):
                    grays.append(p[:4]); st['gray_lowconf'] += 1
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); stem = f.rsplit('.', 1)[0].replace('(', '_').replace(')', '')
    for y in starts(im.height):
        for x in starts(im.width):
            tile(im, x, y, a.tile, objs, grays, f'{stem}_t{x}_{y}', m['split'], st)
    tile(im, 0, 0, max(im.size), objs, grays, f'{stem}_full', m['split'], st)
(OUT / 'flower.yaml').write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: flower\n", encoding='utf-8'); print('EXPORT_DONE', dict(st), flush=True)
