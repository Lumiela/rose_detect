# 시험용: 사람이 보지 않은 모델 예측(의사 라벨)을 더한 탐지 학습 데이터를 라벨 원본과 분리해 만든다.
# 부분 라벨 train 사진: 기존 박스 + 신뢰도 hi 이상 예측은 라벨, lo~hi 예측은 회색으로 가린다. 전체 라벨 사진은 라벨만 쓴다.
# val, test는 data/det를 그대로 연결한다. annotations.csv는 건드리지 않는다.
# 사용: python 48_pseudo_export.py <preds_train.json> [--out data_pseudo] [--hi 0.7] [--lo 0.3]
import argparse, collections as C, json, os, shutil
from PIL import Image, ImageDraw, ImageOps
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('preds'); ap.add_argument('--out', default=str(V11 / 'data_pseudo')); ap.add_argument('--hi', type=float, default=0.7); ap.add_argument('--lo', type=float, default=0.3)
ap.add_argument('--tile', type=int, default=1280); ap.add_argument('--overlap', type=float, default=0.25); ap.add_argument('--imgsz', type=int, default=640); a = ap.parse_args()
OUT = Path(a.out); GRAY = (114, 114, 114); P = json.loads(Path(a.preds).read_text(encoding='utf-8'))
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / 'det' / 'images' / 'train').mkdir(parents=True); (OUT / 'det' / 'labels' / 'train').mkdir(parents=True)
anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(box(r))
for r in read(IGN):
    igns[r['photo']].append(box(r))
rejs = C.defaultdict(list)
for r in read(REJ):
    rejs[r['photo']].append(box(r))


def ov(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    x, y = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (x + y - i + 1e-9), i / min(x, y)


def starts(size):
    if size <= a.tile:
        return [0]
    return list(range(0, size - a.tile, int(a.tile * (1 - a.overlap)))) + [size - a.tile]


def tile(im, x0, y0, side, boxes, ign, stem, st):
    crop = Image.new('RGB', (side, side), GRAY); crop.paste(im.crop((x0, y0, min(x0 + side, im.width), min(y0 + side, im.height))), (0, 0))
    d = ImageDraw.Draw(crop); lines = []
    clip = lambda b: (max(b[0], x0), max(b[1], y0), min(b[2], x0 + side), min(b[3], y0 + side))
    for b in ign:
        c = clip(b)
        if c[0] < c[2] and c[1] < c[3]:
            d.rectangle([c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0], fill=GRAY)
    for b in boxes:
        c = clip(b)
        if not (c[0] < c[2] and c[1] < c[3]):
            continue
        if (c[2] - c[0]) * (c[3] - c[1]) / ((b[2] - b[0]) * (b[3] - b[1])) < 0.5:
            d.rectangle([c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0], fill=GRAY); continue
        lines.append(f'0 {((c[0] + c[2]) / 2 - x0) / side:.6f} {((c[1] + c[3]) / 2 - y0) / side:.6f} {(c[2] - c[0]) / side:.6f} {(c[3] - c[1]) / side:.6f}')
    crop.resize((a.imgsz, a.imgsz), Image.BILINEAR).save(OUT / 'det' / 'images' / 'train' / f'{stem}.jpg', quality=92)
    (OUT / 'det' / 'labels' / 'train' / f'{stem}.txt').write_text('\n'.join(lines), encoding='ascii'); st['tiles'] += 1; st['objects'] += len(lines)


st = C.Counter()
for m in read(MANIFEST):
    if m['split'] != 'train' or m['label_status'] == 'excluded':
        continue
    f = m['photo']; boxes = list(anns[f]); ign = list(igns[f])
    if m['label_status'] == 'partial':
        for p in sorted(P.get(f, []), key=lambda q: -q[4]):
            c = [ov(p[:4], g) for g in boxes]
            if any(i >= 0.3 or s >= 0.6 for i, s in c):
                continue
            if any(i >= 0.3 or s >= 0.6 for i, s in (ov(p[:4], g) for g in rejs[f] + igns[f])):  # 사람이 판정한 영역은 예측으로 덮지 않는다
                st['skipped'] += 1; continue
            if p[4] >= a.hi:
                boxes.append(p[:4]); st['pseudo'] += 1
            elif p[4] >= a.lo and not any(s >= 0.3 for _, s in c):
                ign.append(p[:4]); st['masked'] += 1
    else:
        st['complete_photos'] += 1
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); stem = f.rsplit('.', 1)[0].replace('(', '_').replace(')', '')
    for y in starts(im.height):
        for x in starts(im.width):
            tile(im, x, y, a.tile, boxes, ign, f'{stem}_t{x}_{y}', st)
    tile(im, 0, 0, max(im.size), boxes, ign, f'{stem}_full', st)
for kind in ('images', 'labels'):
    for s in ('val', 'test'):
        os.symlink(V11 / 'data' / 'det' / kind / s, OUT / 'det' / kind / s)
(OUT / 'det' / 'flower.yaml').write_text(f"path: {(OUT / 'det').as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: flower\n", encoding='utf-8')
print(dict(st), flush=True)
