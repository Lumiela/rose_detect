# 사진 단위 평가. v1 42_eval_det.py와 같은 타일, 합치기, 맞춤 규칙이라 점수를 그대로 비교할 수 있다. 라벨은 v1 박스 라벨이다.
# 예측 다각형은 preds_<분할>.json의 6번째 값으로 남긴다. 임계값은 val F1로 정하고 test에 그대로 쓴다.
# 사용: python 31_eval_seg.py <가중치> <출력 폴더> [--splits val,test] [--iou 0.5] [--mask-box]
import argparse, collections as C
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
from ultralytics import YOLO
from infer import detect, iou_ios, open_photo

COMPLETE = {'draft', 'verified'}
ap = argparse.ArgumentParser(); ap.add_argument('weights'); ap.add_argument('out'); ap.add_argument('--splits', default='val,test'); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--iou', type=float, default=0.5); ap.add_argument('--mask-box', action='store_true', help='모델 박스 대신 마스크를 감싼 박스로 평가')
ap.add_argument('--tile', type=int, default=1280); ap.add_argument('--overlap', type=float, default=0.25); ap.add_argument('--imgsz', type=int, default=640); a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); model = YOLO(a.weights)
man = read(MANIFEST); anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def match(f, P, conf):
    gt = anns[f]; G = np.array([box(r) for r in gt]).reshape(-1, 4); P = P[P[:, 4] >= conf]; used = set(); tp = fp = 0
    for p in P:
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2; hit = -1
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
    ph = [m for m in man if m['split'] == split and m['label_status'] in COMPLETE]; preds, polys = {}, {}
    for m in ph:
        preds[m['photo']], polys[m['photo']] = detect(model, open_photo(PHOTOS / m['photo']), a.tile, a.overlap, a.imgsz, a.batch, a.mask_box)
    (OUT / f'preds_{split}.json').write_text(json.dumps({k: [[round(float(x), 3) for x in p] + [np.round(q).astype(int).tolist()] for p, q in zip(v, polys[k])] for k, v in preds.items()}), encoding='utf-8')
    sweep = {c: score(ph, preds, c) for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)}
    if conf is None:
        conf = max(sweep, key=lambda c: sweep[c]['f1'])
    res[split] = dict(chosen=sweep[conf], sweep={str(c): {k: s[k] for k in ('precision', 'recall', 'f1', 'fp', 'fn')} for c, s in sweep.items()})
    print(split, json.dumps(res[split], ensure_ascii=False), flush=True)
(OUT / 'photo_metrics.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); print('EVAL_DONE', flush=True)
