# 탐지와 단계 분류를 연결한 사진 단위 평가. 탐지 결과(preds_<분할>.json)의 박스를 잘라 분류하고 라벨과 맞춘다.
# 맞음은 IoU 기준을 넘고 단계도 같은 경우다. 사진별 단계 개수 오차도 함께 낸다(큐브 상태 확인이 목적이므로).
# 임계값은 탐지 평가에서 val로 정한 값을 쓴다.
# 사용: python 51_eval_e2e.py <탐지 평가 폴더> <분류 가중치> [--splits val,test] [--iou 0.5]
import argparse, collections as C, json, sys
from common import *
sys.path.insert(0, str(V11.parent / 'common'))
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

ap = argparse.ArgumentParser(); ap.add_argument('eval_dir'); ap.add_argument('cls'); ap.add_argument('--splits', default='val,test'); ap.add_argument('--iou', type=float, default=0.5); a = ap.parse_args()
E = Path(a.eval_dir); conf = json.loads((E / 'photo_metrics.json').read_text(encoding='utf-8'))['val']['chosen']['conf']
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
net = efficientnet_v2_s(); net.classifier[1] = nn.Linear(net.classifier[1].in_features, 3); net.load_state_dict(torch.load(a.cls, map_location='cpu')); net = net.cuda().eval()
man = read(MANIFEST); anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def classify(f, P):
    if not len(P):
        return []
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); xs = []
    for p in P:
        s = max(p[2] - p[0], p[3] - p[1]) * 1.25; cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        xs.append(TF(im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))))
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(xs), 128):
            out += net(torch.stack(xs[i:i + 128]).cuda()).float().argmax(1).tolist()
    return out


def iou(b, g):
    w = max(0, min(b[2], g[2]) - max(b[0], g[0])); h = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = w * h
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i + 1e-9)


def pairs(f, P, Y):
    # (라벨 단계, 예측 단계) 목록. 3은 없음(놓침 또는 오탐). 제외 영역 안의 오탐은 세지 않고 개수에도 넣지 않는다
    gt = anns[f]; used = set(); out = []
    for p, y in zip(P, Y):
        v = [iou(p, box(r)) if j not in used else 0 for j, r in enumerate(gt)]; j = int(np.argmax(v)) if v else -1
        if j >= 0 and v[j] >= a.iou:
            used.add(j); out.append((STAGES.index(gt[j]['stage']), y)); continue
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in igns[f]):
            out.append((3, y))
    return out + [(STAGES.index(r['stage']), 3) for j, r in enumerate(gt) if j not in used]


def summary(items):
    cm = np.zeros((4, 4), int)
    for t, q in items:
        cm[t, q] += 1
    rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(3)]; pre = [cm[i, i] / max(cm[:, i].sum(), 1) for i in range(3)]
    f1 = [2 * r * q / max(r + q, 1e-9) for r, q in zip(rec, pre)]
    return dict(recall=[round(float(v), 4) for v in rec], precision=[round(float(v), 4) for v in pre], f1=[round(float(v), 4) for v in f1], macro_f1=round(float(np.mean(f1)), 4), confusion=cm.tolist())


res = dict(conf=conf, cls=a.cls)
for split in a.splits.split(','):
    D = json.loads((E / f'preds_{split}.json').read_text(encoding='utf-8')); by = {}; cnt = []
    for f, ps in D.items():
        P = [p[:4] for p in ps if p[4] >= conf]; Y = classify(f, P); by[f] = pairs(f, P, Y)
        g = [sum(t == i for t, _ in by[f]) for i in range(3)]; q = [sum(y == i for _, y in by[f]) for i in range(3)]
        cnt.append(dict(photo=f, label=g, pred=q))
    s = summary([t for v in by.values() for t in v]); ph = list(by); rng = np.random.default_rng(0)
    s['macro_f1_photo_ci95'] = [round(float(v), 4) for v in np.percentile([summary([t for k in rng.integers(len(ph), size=len(ph)) for t in by[ph[k]]])['macro_f1'] for _ in range(1000)], [2.5, 97.5])]
    L, Q = np.array([c['label'] for c in cnt]), np.array([c['pred'] for c in cnt])
    s['count_total_label'] = L.sum(0).tolist(); s['count_total_pred'] = Q.sum(0).tolist(); s['count_mae_per_photo'] = [round(float(v), 2) for v in np.abs(L - Q).mean(0)]; s['per_photo'] = cnt
    res[split] = s; print(split, json.dumps({k: v for k, v in s.items() if k != 'per_photo'}, ensure_ascii=False), flush=True)
(E / 'e2e_metrics.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); print('E2E_DONE', flush=True)
