# 라벨 초안(labels/drafts/<시각>.json)을 라벨 원본에 반영한다. 해당 사진의 박스는 초안으로 통째로 바뀐다.
# 초안 형식: {"photo": 이름, "scale": 0.5, "parts": {"0": {"boxes": [[x0,y0,x1,y1,"B"],...], "ignore": [[x0,y0,x1,y1],...]}, "1": {...}}}
#   좌표는 10_view_photo.py가 만든 조각 이미지 기준. B=봉오리, O=개화중, F=만개
# 사용: python 11_apply_draft.py [시각 ...] (생략하면 drafts 전체)
import json, sys
from common import *

CODE = {'B': '봉오리', 'O': '개화중', 'F': '만개'}
files = sorted((V11 / 'labels' / 'drafts').glob('*.json'))
if sys.argv[1:]:
    files = [p for p in files if any(k in p.name for k in sys.argv[1:])]
ann, ign, man = read(ANN), read(IGN), read(MANIFEST)
info = {m['photo']: m for m in man}
for p in files:
    d = json.loads(p.read_text(encoding='utf-8')); f = d['photo']; m = info[f]; s = d.get('scale', 0.5)
    W, H = int(m['width']), int(m['height'])
    offs = {'0': (0, 0), '1': (0, H // 2 - 60) if H >= W else (W // 2 - 60, 0)}
    ann = [r for r in ann if r['photo'] != f]; ign = [r for r in ign if r['photo'] != f]
    n = 0
    for part, c in d['parts'].items():
        ox, oy = offs[part]
        conv = lambda b: [min(W, max(0, round(b[0] / s + ox))), min(H, max(0, round(b[1] / s + oy))),
                          min(W, max(0, round(b[2] / s + ox))), min(H, max(0, round(b[3] / s + oy)))]
        for b in c.get('boxes', []):
            n += 1; x0, y0, x1, y1 = conv(b)
            assert x0 < x1 and y0 < y1, (f, b)
            ann.append(dict(photo=f, ann_id=f'D{n:03d}', x0=x0, y0=y0, x1=x1, y1=y1, stage=CODE[b[4]], source='draft'))
        for b in c.get('ignore', []):
            x0, y0, x1, y1 = conv(b); assert x0 < x1 and y0 < y1, (f, b)
            ign.append(dict(photo=f, x0=x0, y0=y0, x1=x1, y1=y1))
    if d.get('exclude'):  # 온실 전경처럼 운영 시점과 다른 사진: 내보내기에서 뺀다
        m['label_status'] = 'excluded'
    elif m['label_status'] != 'verified':
        m['label_status'] = 'draft'
    print(f, 'boxes', n, 'ignore', sum(r['photo'] == f for r in ign))
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); write(MANIFEST, MAN_FIELDS, man)
