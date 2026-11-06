# ESP 사진 평가. 찾음은 예측 상자의 중심이 사람이 그린 상자 안에 있는 경우이고, 봉오리 하나에 한 번만 센다. 다른 예측은 중심이 어떤 사람 상자에도 없는 예측이다. IoU 0.5는 몸통 상자와 비교해 함께 적는다.
# val: 임계값을 정해 selected.json에 고정한다. test: 고정한 값으로 한 번만 잰다. 사용: python 31_esp_eval.py val | test
import os, sys
from common import *
sys.path.append(str(V3 / 'scripts'))
from infer import detect, open_photo
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
split = sys.argv[1]; OUT = VER / 'runs' / 'eval'; OUT.mkdir(parents=True, exist_ok=True); SEL = OUT / 'selected.json'; SP = splits(); PH = sorted(p for p, s in SP.items() if s == split)
LAB = {}
for t in (VER / 'labels' / 'masks_body.jsonl').read_text(encoding='utf-8').splitlines():
    r = json.loads(t); LAB.setdefault(r['photo'], []).append(r)
A = str(V3 / 'runs/seg/A_base/weights/best.pt'); W = {'v3_t1280': (A, 1280), 'v3_t880': (A, 880)}
for s in (0, 1, 2):
    W[f'ft_s{s}_t880'] = (str(VER / f'runs/seg/ft_s{s}/weights/best.pt'), 880)


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0 if w <= 0 or h <= 0 else w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def score(pred, thr):
    hit = 0; other = 0; h50 = 0; n = 0; buds = {}; per = {}
    for p in PH:
        P = [q for q in pred[p] if q[4] >= thr]; L = LAB[p]; used = set(); per[p[:6]] = dict(n=len(P), found=[], other=[])
        for q in sorted(P, key=lambda q: -q[4]):
            cx, cy = (q[0] + q[2]) / 2, (q[1] + q[3]) / 2; c = [i for i, r in enumerate(L) if r['box'][0] <= cx <= r['box'][2] and r['box'][1] <= cy <= r['box'][3]]
            if not c:
                other += 1; per[p[:6]]['other'].append([round(cx), round(cy), round(q[4], 2)]); continue
            i = c[0]
            if i in used:
                continue
            used.add(i); hit += 1; h50 += iou(q[:4], L[i]['tight']) >= .5; per[p[:6]]['found'].append([L[i]['bud'], round(q[4], 2)])
        n += len(L)
        for i, r in enumerate(L):
            buds.setdefault(r['bud'], [0, 0]); buds[r['bud']][1] += 1; buds[r['bud']][0] += i in used
    return dict(found=hit, total=n, other=other, iou50=int(h50), buds=buds, per=per)


pred = {}
for name, (w, tile) in W.items():
    m = YOLO(w); pred[name] = {}
    for p in PH:
        P, _ = detect(m, open_photo(PHOTOS / p), tile=tile); pred[name][p] = [[float(x) for x in q] for q in P]
    del m
(OUT / f'preds_{split}.json').write_text(json.dumps(pred), encoding='utf-8')
THR = [round(.05 * i, 2) for i in range(2, 17)]
if split == 'val':
    sel = {}
    for name in W:
        sw = {t: score(pred[name], t) for t in THR}; best = max(THR, key=lambda t: (sw[t]['found'] - sw[t]['other'], t)); sel[name] = best
        print(name, 'val 선택', best, {t: (sw[t]['found'], sw[t]['other']) for t in THR}, flush=True)
    SEL.write_text(json.dumps(sel), encoding='utf-8'); sys.exit()
sel = json.loads(SEL.read_text(encoding='utf-8')); res = {}
for name in W:
    t = .6 if name.startswith('v3') else sel[name]; res[name] = dict(thr=t, **score(pred[name], t)); print(name, 'thr', t, {k: res[name][k] for k in ('found', 'total', 'other', 'iou50', 'buds')}, res[name]['per'], flush=True)
    if name.startswith('v3'):
        tt = sel[name]; rr = score(pred[name], tt); print('  참고 val 임계값', tt, rr['found'], rr['other'], flush=True)
(OUT / 'test_result.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
