# 2단계: 꽃 크롭을 EfficientNetV2-S로 봉오리/개화중/만개 분류. 잎 v8과 같은 학습법에 클래스 가중치를 더했다.
# val 매크로 F1이 가장 높은 가중치를 고르고, 분할별 전체와 구간(색), 기기별 점수를 metrics.json에 남긴다.
# 사용: python 40_train_cls.py [--epochs 20] [--name effnetv2s]
import argparse, collections as C, csv, json, sys, time
from common import *
sys.path.insert(0, str(V11.parent / 'common'))
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--name', default='effnetv2s'); ap.add_argument('--eval-only', default='', help='학습 없이 best.pt를 평가하고 결과 이름에 붙일 표시')
a = ap.parse_args()
DATA = V11 / 'data' / 'cls'; out = V11 / 'runs' / 'cls' / a.name; out.mkdir(parents=True, exist_ok=True)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = (224, 224)
TF_TRAIN = T.Compose([T.Resize(SIZE), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.1, 0.1, 0.1, 0.01), T.ToTensor(), T.Normalize(MEAN, STD)])
TF_EVAL = T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
rows = read(DATA / 'index.csv')
for r in rows:
    r['y'] = STAGES.index(r['stage'])


def of(split):
    return [r for r in rows if r['split'] == split]


class DS(torch.utils.data.Dataset):
    def __init__(self, it, tf): self.it, self.tf = it, tf
    def __len__(self): return len(self.it)
    def __getitem__(self, i):
        r = self.it[i]; return self.tf(Image.open(DATA / r['file']).convert('RGB')), r['y']


def probs(model, it):
    dl = torch.utils.data.DataLoader(DS(it, TF_EVAL), batch_size=128, shuffle=False, num_workers=4)
    model.eval(); P = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for x, _ in dl:
            P.append(model(x.cuda()).float().softmax(1).cpu())
    return torch.cat(P).numpy()


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p); cm = np.zeros((3, 3), int)
    np.add.at(cm, (y, p), 1)
    rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(3)]; pre = [cm[i, i] / max(cm[:, i].sum(), 1) for i in range(3)]
    f1 = [2 * r * q / max(r + q, 1e-9) for r, q in zip(rec, pre)]; seen = [i for i in range(3) if cm[i].sum()]
    return dict(n=len(y), acc=round(float((y == p).mean()), 4), macro_f1=round(float(np.mean([f1[i] for i in seen])), 4), recall=[round(float(v), 4) for v in rec],
                precision=[round(float(v), 4) for v in pre], f1=[round(float(v), 4) for v in f1], confusion=cm.tolist())


def photo_ci(it, pred, n=1000):
    # 한 사진의 꽃은 서로 독립이 아니므로 사진 단위로 다시 뽑아 구간을 낸다
    by = C.defaultdict(list)
    for r, q in zip(it, pred):
        by[r['photo']].append((r['y'], q))
    ph, rng, vals = list(by), np.random.default_rng(0), []
    for _ in range(n):
        s = [t for k in rng.integers(len(ph), size=len(ph)) for t in by[ph[k]]]
        vals.append(metrics([t[0] for t in s], [t[1] for t in s])['macro_f1'])
    return [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]


torch.manual_seed(42)
model = efficientnet_v2_s()
model.load_state_dict(torch.load(paths.WEIGHTS / 'torch_hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth', map_location='cpu'))
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3); model = model.cuda()
tr_items, va_items = of('train'), of('val')
cnt = np.bincount([r['y'] for r in tr_items], minlength=3); weight = torch.tensor(len(tr_items) / (3 * cnt), dtype=torch.float32).cuda()
tr = torch.utils.data.DataLoader(DS(tr_items, TF_TRAIN), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.epochs * len(tr), pct_start=0.1)
scaler, crit, best = torch.cuda.amp.GradScaler(), nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1), -1
print('train', cnt.tolist(), 'val', len(va_items), 'weight', weight.tolist(), flush=True)
t0 = time.time()
for ep in range(0 if a.eval_only else a.epochs):
    model.train(); tl = n = 0
    for x, y in tr:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(y); n += len(y)
    P = probs(model, va_items); m = metrics([r['y'] for r in va_items], P.argmax(1))
    if m['macro_f1'] > best:
        best = m['macro_f1']; torch.save(model.state_dict(), out / 'best.pt')
    print(f"epoch {ep + 1}/{a.epochs} loss {tl / n:.4f} val acc {m['acc']} macroF1 {m['macro_f1']} recall {m['recall']} (best {best}) {time.time() - t0:.0f}s", flush=True)
model.load_state_dict(torch.load(out / 'best.pt')); res = {}
for split in ('val', 'test'):
    it = of(split); P = probs(model, it); pred = P.argmax(1); y = [r['y'] for r in it]
    res[split] = metrics(y, pred); res[split]['macro_f1_photo_ci95'] = photo_ci(it, pred)
    for key in ('block', 'device'):
        res[split]['by_' + key] = {g: metrics([r['y'] for r in it if r[key] == g], [q for r, q in zip(it, pred) if r[key] == g]) for g in sorted({r[key] for r in it})}
    with open(out / f'probs_{split}{a.eval_only}.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['file', 'photo', 'block', 'stage', 'pred'] + STAGES)
        w.writerows([[r['file'], r['photo'], r['block'], r['stage'], STAGES[q]] + [round(float(v), 4) for v in pr] for r, q, pr in zip(it, pred, P)])
    print(split, json.dumps(res[split], ensure_ascii=False), flush=True)
(out / f'metrics{a.eval_only}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); print('TRAIN_DONE', flush=True)
