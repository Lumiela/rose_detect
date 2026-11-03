# 사람 검수용 이미지: 사진마다 박스, 번호, 단계 색을 그린다. 회색 빗금 없이 보라색 테두리는 제외 영역이다.
# 사용: python 12_review_sheets.py [분할 이름, 기본 test]
import collections as C, sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
from common import *

split = sys.argv[1] if len(sys.argv) > 1 else 'test'
OUT = VER / 'sheets' / f'review_{split}'; OUT.mkdir(parents=True, exist_ok=True)
COL = {'봉오리': (255, 40, 40), '개화중': (255, 220, 0), '만개': (0, 220, 255)}
anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))
font = ImageFont.load_default(size=40); small = ImageFont.truetype('malgun.ttf', 60)
rows = []
for m in read(MANIFEST):
    if m['split'] != split or m['label_status'] not in COMPLETE:
        continue
    f = m['photo']; im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); d = ImageDraw.Draw(im)
    for b in igns[f]:
        d.rectangle(b, outline=(200, 0, 255), width=7)
    for r in anns[f]:
        b = box(r); d.rectangle(b, outline=COL[r['stage']], width=7)
        d.text((b[0], max(0, b[1] - 44)), r['ann_id'][1:].lstrip('0'), fill=COL[r['stage']], font=font, stroke_width=4, stroke_fill=(0, 0, 0))
    c = C.Counter(r['stage'] for r in anns[f])
    d.text((20, 20), f"{f}  봉오리 {c['봉오리']} / 개화중 {c['개화중']} / 만개 {c['만개']}", fill=(255, 255, 255), font=small, stroke_width=5, stroke_fill=(0, 0, 0))
    s = 2000 / max(im.size); im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS).save(OUT / f, quality=88)
    rows.append((f, len(anns[f]), c)); print(f, len(anns[f]), dict(c))
print('photos', len(rows), 'boxes', sum(r[1] for r in rows), sum((r[2] for r in rows), C.Counter()))
