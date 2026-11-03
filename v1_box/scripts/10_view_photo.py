# 라벨 작업용 보기: 사진을 위아래(또는 좌우) 두 조각으로 나눠 격자와 현재 박스를 그린다.
# 사용: python 10_view_photo.py <사진 이름 또는 시각> [출력 폴더] [--scale 0.5]
import argparse, sys
from PIL import Image, ImageOps, ImageDraw, ImageFont
from common import *

COL = {'봉오리': (255, 40, 40), '개화중': (255, 220, 0), '만개': (0, 220, 255)}
ap = argparse.ArgumentParser(); ap.add_argument('photo'); ap.add_argument('out', nargs='?', default=str(V11 / 'sheets' / 'work'))
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--grid', type=int, default=250)
a = ap.parse_args()
names = [p.name for p in PHOTOS.glob('*.jpg')]
f = a.photo if a.photo in names else next(n for n in sorted(names) if a.photo in n)
anns = [r for r in read(ANN) if r['photo'] == f]; ign = [r for r in read(IGN) if r['photo'] == f]
im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB')
W, H = im.size; d = ImageDraw.Draw(im)
font = ImageFont.load_default(size=34)
for r in ign:
    d.rectangle(box(r), outline=(200, 0, 255), width=6)
for r in anns:
    b = box(r); d.rectangle(b, outline=COL[r['stage']], width=5)
    d.text((b[0], max(0, b[1] - 38)), r['ann_id'], fill=COL[r['stage']], font=font, stroke_width=3, stroke_fill=(0, 0, 0))
for x in range(0, W, a.grid):
    d.line([(x, 0), (x, H)], fill=(255, 255, 255) if x % 1000 == 0 else (190, 190, 190), width=2 if x % 1000 == 0 else 1)
for y in range(0, H, a.grid):
    d.line([(0, y), (W, y)], fill=(255, 255, 255) if y % 1000 == 0 else (190, 190, 190), width=2 if y % 1000 == 0 else 1)
for x in range(0, W, a.grid * 2):
    for y in range(0, H, a.grid * 2):
        d.text((x + 4, y + 2), f'{x},{y}', fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
Path(a.out).mkdir(parents=True, exist_ok=True)
parts = [(0, 0, W, H // 2 + 60), (0, H // 2 - 60, W, H)] if H >= W else [(0, 0, W // 2 + 60, H), (W // 2 - 60, 0, W, H)]
stem = f.rsplit('.', 1)[0]
for i, p in enumerate(parts):
    c = im.crop(p); c = c.resize((int(c.width * a.scale), int(c.height * a.scale)), Image.LANCZOS)
    c.save(Path(a.out) / f'{stem}_{i}.jpg', quality=88)
print(f, (W, H), 'boxes', len(anns), 'ignore', len(ign), [str(Path(a.out) / f'{stem}_{i}.jpg') for i in range(2)])
