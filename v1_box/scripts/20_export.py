# 라벨 원본에서 1단계(탐지)와 2단계(분류) 학습 데이터를 만든다. 라벨링을 끝낸 사진(draft, verified)만 탐지에 쓴다.
# 1단계: YOLO 형식 data/det/images|labels/<split>, 클래스 1개(flower)
#   세밀(1280px 타일을 640으로)과 전체(사진 전체를 640으로) 두 축척. 잘린 꽃은 보이는 면적이 50% 미만이면 회색으로 가린다.
# 2단계: data/cls/<split>/<단계>/, 박스 긴 변 × 1.25 정사각형 크롭(leafdx 2단계와 같은 방식), 이미지 밖은 검은색
#   박스가 있는 사진은 전부 쓴다(partial 포함). source 열로 기존 라벨(legacy)을 구분한다.
# 사용: python 20_export.py [--out data] [--splits train,val,test] [--limit N]
import argparse, collections as C, json, shutil
from PIL import Image, ImageDraw, ImageOps
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(VER / 'data')); ap.add_argument('--splits', default='train,val,test')
ap.add_argument('--tile', type=int, default=1280); ap.add_argument('--overlap', type=float, default=0.25); ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--limit', type=int, default=0, help='시험용: 분할마다 사진 수 제한'); a = ap.parse_args()
OUT = Path(a.out); splits = a.splits.split(',')
if OUT.exists():
    shutil.rmtree(OUT)
GRAY = (114, 114, 114)
man = [m for m in read(MANIFEST) if m['split'] in splits and m['label_status'] != 'excluded']
anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def starts(size, tile, step):
    if size <= tile:
        return [0]
    s = list(range(0, size - tile, step)) + [size - tile]
    return s


def det_tile(im, x0, y0, side, boxes, ign, stem, split, stats):
    """정사각 타일 하나를 저장. 라벨이 없는 타일도 음성 예시로 저장한다."""
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
        vis = (c[2] - c[0]) * (c[3] - c[1]) / ((b[2] - b[0]) * (b[3] - b[1]))
        if vis < 0.5:
            d.rectangle([c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0], fill=GRAY); stats['clipped_masked'] += 1
            continue
        cx, cy, w, h = ((c[0] + c[2]) / 2 - x0) / side, ((c[1] + c[3]) / 2 - y0) / side, (c[2] - c[0]) / side, (c[3] - c[1]) / side
        lines.append(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    (OUT / 'det' / 'images' / split).mkdir(parents=True, exist_ok=True); (OUT / 'det' / 'labels' / split).mkdir(parents=True, exist_ok=True)
    crop.resize((a.imgsz, a.imgsz), Image.BILINEAR).save(OUT / 'det' / 'images' / split / f'{stem}.jpg', quality=92)
    (OUT / 'det' / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines), encoding='ascii')
    stats['tiles'] += 1; stats['objects'] += len(lines); stats['empty_tiles'] += not lines


stat = C.defaultdict(C.Counter); cls_rows = []; seen = C.Counter()
for m in man:
    if a.limit and seen[m['split']] >= a.limit:
        continue
    seen[m['split']] += 1
    f, split = m['photo'], m['split']; rs = anns[f]; complete = m['label_status'] in COMPLETE
    if not complete and not rs:
        continue
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); stem = f.rsplit('.', 1)[0].replace('(', '_').replace(')', '')
    if complete:
        boxes = [box(r) for r in rs]; step = int(a.tile * (1 - a.overlap))
        for y in starts(im.height, a.tile, step):
            for x in starts(im.width, a.tile, step):
                det_tile(im, x, y, a.tile, boxes, igns[f], f'{stem}_t{x}_{y}', split, stat[split])
        det_tile(im, 0, 0, max(im.size), boxes, igns[f], f'{stem}_full', split, stat[split])
    for r in rs:
        b = box(r); side = max(b[2] - b[0], b[3] - b[1]) * 1.25; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        p = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
        d = OUT / 'cls' / split / r['stage']; d.mkdir(parents=True, exist_ok=True)
        name = f"{stem}_{r['ann_id']}.jpg"; im.crop(p).save(d / name, quality=92)
        cls_rows.append(dict(file=f"{split}/{r['stage']}/{name}", split=split, stage=r['stage'], source=r['source'], photo=f, block=m['block'], device=m['device']))
        stat[split]['cls_' + r['stage']] += 1
if cls_rows:
    write(OUT / 'cls' / 'index.csv', list(cls_rows[0]), cls_rows)
names = {s: f'images/{s}' for s in splits if (OUT / 'det' / 'images' / s).exists()}
(OUT / 'det' / 'flower.yaml').write_text(
    'path: ' + str((OUT / 'det').as_posix()) + '\n' + ''.join(f'{k}: {v}\n' for k, v in names.items()) + 'names:\n  0: flower\n', encoding='utf-8')
print(json.dumps({k: dict(v) for k, v in stat.items()}, ensure_ascii=False, indent=1))
