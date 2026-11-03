# 박스 라벨과 임시 라벨을 프롬프트로 SAM2.1이 꽃 마스크를 만든다. 단계와 상관없이 모든 꽃을 만든다.
# 사진 전체를 1024로 줄이면 작은 봉오리가 뭉개지므로 박스 주변 창을 잘라 넣는다. 사람이 보지 않은 초안이다.
# 임시 라벨은 v1 48_pseudo_export.py와 같은 규칙으로 고른다 (신뢰도 hi 이상, 기존 라벨, 제외 영역, 차단 목록과 겹치지 않음).
# --retry: 박스 IoU 0.5 미만이거나 마스크가 없는 항목만 창과 박스를 바꿔 다시 만든다. 그래도 실패하면 use=false로 표시해 학습에서 가린다.
# 사용: python 10_make_masks.py [--out 파일] [--splits train,val,test] [--limit 사진 수] [--hi 0.7] [--retry]
import argparse, collections as C, time
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import SAM

ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(MASKS)); ap.add_argument('--splits', default='train,val,test'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--hi', type=float, default=0.7)
ap.add_argument('--pad', type=float, default=0.75, help='박스 긴 변 대비 창 여백'); ap.add_argument('--retry', action='store_true'); a = ap.parse_args()
model = SAM(str(paths.WEIGHTS / 'sam2.1_s.pt')); P = json.loads(PSEUDO_PREDS.read_text(encoding='utf-8'))
anns, igns, rejs = C.defaultdict(list), C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))
for r in read(REJ):
    rejs[r['photo']].append(box(r))


def targets(m):
    f = m['photo']; out = [dict(key=r['ann_id'], stage=r['stage'], source=r['source'], box=box(r)) for r in anns[f]]
    if m['label_status'] != 'partial':
        return out
    for k, p in enumerate(sorted(P.get(f, []), key=lambda q: -q[4])):
        if p[4] < a.hi or any(i >= 0.3 or s >= 0.6 for i, s in (ov(p[:4], t['box']) for t in out)):
            continue
        if any(i >= 0.3 or s >= 0.6 for i, s in (ov(p[:4], g) for g in rejs[f] + igns[f])):
            continue
        out.append(dict(key=f'Q{k:04d}', stage='', source='pseudo', box=[round(v, 1) for v in p[:4]]))
    return out


def segment(img, b, pad_ratio=None):
    H, W = img.shape[:2]; pad = max(b[2] - b[0], b[3] - b[1]) * (pad_ratio or a.pad) + 16
    x0, y0, x1, y1 = max(0, int(b[0] - pad)), max(0, int(b[1] - pad)), min(W, int(b[2] + pad + 1)), min(H, int(b[3] + pad + 1))
    crop = np.ascontiguousarray(img[y0:y1, x0:x1, ::-1])
    r = model(crop, bboxes=[[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]], verbose=False)[0]
    if r.masks is None or not len(r.masks.data):
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8); sc = float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes) else 0.0
    if m.shape != crop.shape[:2]:
        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); m = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + [x0, y0]
    if len(poly) < 3:
        return None
    ys, xs = np.nonzero(m); mb = [int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0 + 1), int(ys.max() + y0 + 1)]
    area = int(m.sum()); inside = int(m[max(0, int(b[1]) - y0):int(b[3]) - y0 + 1, max(0, int(b[0]) - x0):int(b[2]) - x0 + 1].sum())
    return dict(score=round(sc, 3), area=area, mask_box=mb, box_iou=round(ov(mb, b)[0], 3), fill=round(area / ((b[2] - b[0]) * (b[3] - b[1]) + 1e-9), 3), outside=round(1 - inside / max(area, 1), 3), parts=int(n - 1), poly=poly.tolist())


def retry():
    rows = read_masks(a.out); n = C.Counter()
    by = C.defaultdict(list)
    for r in rows:
        if not r['poly'] or r['box_iou'] < 0.5:
            by[r['photo']].append(r)
    for f, rs in by.items():
        img = np.asarray(ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'))
        for r in rs:
            b = r['box']; w, h = b[2] - b[0], b[3] - b[1]; best = None
            for pr, g in [(0.3, 0), (1.5, 0), (3.0, 0), (0.75, 0.1), (1.5, 0.1), (0.75, -0.1)]:
                q = segment(img, [b[0] - w * g, b[1] - h * g, b[2] + w * g, b[3] + h * g], pr)
                if q is None:
                    continue
                q['box_iou'] = round(ov(q['mask_box'], b)[0], 3); q['fill'] = round(q['area'] / (w * h + 1e-9), 3)
                if best is None or q['box_iou'] > best['box_iou']:
                    best = q
            if best is not None and best['box_iou'] >= 0.5 and best['fill'] >= 0.25:
                r.update(best); r['retry'] = True; n['fixed'] += 1
            else:
                r['use'] = False; n['failed'] += 1
    Path(a.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8'); print('RETRY_DONE', dict(n), flush=True)


if a.retry:
    retry(); sys.exit()
ph = [m for m in read(MANIFEST) if m['split'] in a.splits.split(',') and m['label_status'] != 'excluded']
if a.limit:
    ph = ph[:a.limit]
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); n = C.Counter()
with open(out.with_suffix('.tmp'), 'w', encoding='utf-8') as f:
    for i, m in enumerate(ph):
        img = np.asarray(ImageOps.exif_transpose(Image.open(PHOTOS / m['photo'])).convert('RGB'))
        for t in targets(m):
            q = segment(img, t['box']); n[m['split']] += 1; n['empty'] += q is None
            f.write(json.dumps(dict(photo=m['photo'], split=m['split'], **t, **(q or dict(poly=None))), ensure_ascii=False) + '\n')
        f.flush(); print(f"{i + 1}/{len(ph)} {m['photo']} {dict(n)} {time.time() - t0:.0f}s", flush=True)
out.with_suffix('.tmp').replace(out); print('MASK_DONE', dict(n), flush=True)
