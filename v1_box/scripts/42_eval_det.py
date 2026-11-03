# 사진 단위 탐지 평가. 운영과 같이 타일과 전체 축척으로 찾아 원본 좌표로 합친 뒤 라벨과 맞춘다.
# 임계값은 val F1로 정하고 test에 그대로 쓴다. 예측은 preds_<분할>.json으로 남겨 라벨 후보와 단계 분류 연결에 쓴다.
# 사용: python 42_eval_det.py <가중치> <출력 폴더> [--splits val,test] [--batch 32]
import argparse, collections as C, json, sys
from common import *
sys.path.insert(0, str(ROOT / 'common'))
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument('weights'); ap.add_argument('out'); ap.add_argument('--splits', default='val,test'); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--iou', type=float, default=0.5); ap.add_argument('--all-status', action='store_true', help='부분 라벨 사진까지 예측 (라벨 후보용)'); ap.add_argument('--keep-fragments', action='store_true', help='타일 경계 조각을 거르지 않음 (비교용)')
ap.add_argument('--tile', type=int, default=1280); ap.add_argument('--overlap', type=float, default=0.25); ap.add_argument('--imgsz', type=int, default=640); a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); model = YOLO(a.weights)
man = read(MANIFEST); anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def starts(size):
    if size <= a.tile:
        return [0]
    return list(range(0, size - a.tile, int(a.tile * (1 - a.overlap)))) + [size - a.tile]


def iou_ios(b, B):
    x0, y0 = np.maximum(b[0], B[:, 0]), np.maximum(b[1], B[:, 1]); x1, y1 = np.minimum(b[2], B[:, 2]), np.minimum(b[3], B[:, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None); ab = (b[2] - b[0]) * (b[3] - b[1]); aB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return i / (ab + aB - i + 1e-9), i / (np.minimum(ab, aB) + 1e-9)


def merge(P):
    # 겹친 타일의 중복과, 타일 경계에서 잘려 큰 박스 안에 들어간 조각을 뺀다
    # P 열: x0 y0 x1 y1 conf 조각. 조각은 사진 가장자리가 아닌 타일 경계에 닿은 박스다
    whole = P[P[:, 5] == 0, :4]
    if not a.keep_fragments and len(whole):
        P = np.array([p for p in P if not p[5] or iou_ios(p[:4], whole)[1].max() < 0.5]).reshape(-1, 6)
    P = P[np.argsort(-P[:, 4]), :5]; keep = []
    for p in P:
        if keep:
            io, os_ = iou_ios(p[:4], np.array(keep)[:, :4])
            if (io > 0.5).any() or (os_ > 0.8).any():
                continue
        keep.append(p)
    return np.array(keep).reshape(-1, 5)


def detect(f):
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); views = []; side = max(im.size)
    for y in starts(im.height):
        for x in starts(im.width):
            c = Image.new('RGB', (a.tile, a.tile), (114, 114, 114)); c.paste(im.crop((x, y, min(x + a.tile, im.width), min(y + a.tile, im.height))), (0, 0))
            views.append((c.resize((a.imgsz, a.imgsz), Image.BILINEAR), x, y, a.tile / a.imgsz))
    c = Image.new('RGB', (side, side), (114, 114, 114)); c.paste(im, (0, 0)); views.append((c.resize((a.imgsz, a.imgsz), Image.BILINEAR), 0, 0, side / a.imgsz))
    P = []
    for i in range(0, len(views), a.batch):
        vs = views[i:i + a.batch]
        for v, r in zip(vs, model.predict([t[0] for t in vs], imgsz=a.imgsz, conf=0.05, iou=0.6, verbose=False)):
            t = r.boxes.xyxy.cpu().numpy(); b = t * v[3]; b[:, [0, 2]] += v[1]; b[:, [1, 3]] += v[2]
            full = v[3] != a.tile / a.imgsz; m = 2
            frag = np.zeros(len(b), bool) if full else ((t[:, 0] <= m) & (v[1] > 0)) | ((t[:, 1] <= m) & (v[2] > 0)) | ((t[:, 2] >= a.imgsz - m) & (v[1] + a.tile < im.width)) | ((t[:, 3] >= a.imgsz - m) & (v[2] + a.tile < im.height))
            P.append(np.c_[b, r.boxes.conf.cpu().numpy(), frag])
    P = merge(np.vstack(P)) if P else np.zeros((0, 5))
    P[:, [0, 2]] = P[:, [0, 2]].clip(0, im.width); P[:, [1, 3]] = P[:, [1, 3]].clip(0, im.height)
    return P


def match(f, P, conf):
    # (TP, FP, 놓친 라벨 목록). 제외 영역 안에 중심이 있는 예측은 세지 않는다
    gt = anns[f]; G = np.array([box(r) for r in gt]).reshape(-1, 4); P = P[P[:, 4] >= conf]; used = set(); tp = fp = 0
    for p in P:
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        hit = -1
        if len(G):
            io = iou_ios(p[:4], G)[0]; io[list(used)] = 0; j = int(io.argmax())
            if io[j] >= a.iou:
                hit = j
        if hit >= 0:
            used.add(hit); tp += 1
        elif not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in igns[f]):
            fp += 1
    return tp, fp, [gt[j] for j in range(len(gt)) if j not in used]


def bucket(r):
    b = box(r); s = max(b[2] - b[0], b[3] - b[1])
    return '<80px' if s < 80 else '80-160px' if s < 160 else '>=160px'


def score(ph, preds, conf):
    tp = fp = 0; gt_n, miss = C.Counter(), C.Counter(); per = []
    for m in ph:
        f = m['photo']; t, q, ms = match(f, preds[f], conf); tp += t; fp += q; n = len(anns[f])
        per.append(dict(photo=f, block=m['block'], gt=n, tp=t, fp=q))
        for r in anns[f]:
            for k in (r['stage'], bucket(r), m['block'], m['device']):
                gt_n[k] += 1
        for r in ms:
            for k in (r['stage'], bucket(r), m['block'], m['device']):
                miss[k] += 1
    fn = sum(len(anns[m['photo']]) for m in ph) - tp; pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(conf=conf, tp=tp, fp=fp, fn=fn, precision=round(pre, 4), recall=round(rec, 4), f1=round(2 * pre * rec / max(pre + rec, 1e-9), 4),
                recall_by={k: [round(1 - miss[k] / gt_n[k], 4), gt_n[k]] for k in sorted(gt_n)}, per_photo=per)


res = {}; conf = None
for split in a.splits.split(','):
    ph = [m for m in man if m['split'] == split and m['label_status'] != 'excluded' and (a.all_status or m['label_status'] in COMPLETE)]; preds = {m['photo']: detect(m['photo']) for m in ph}
    (OUT / f'preds_{split}.json').write_text(json.dumps({k: np.round(v, 3).tolist() for k, v in preds.items()}), encoding='utf-8')
    sweep = {c: score(ph, preds, c) for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)}
    if conf is None:
        conf = max(sweep, key=lambda c: sweep[c]['f1'])
    res[split] = dict(chosen=sweep[conf], sweep={str(c): {k: s[k] for k in ('precision', 'recall', 'f1', 'fp', 'fn')} for c, s in sweep.items()})
    print(split, json.dumps(res[split], ensure_ascii=False), flush=True)
(OUT / 'photo_metrics.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); print('EVAL_DONE', flush=True)
