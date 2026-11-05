# 꽃 모델 v1, v2, v3를 ESP 카메라 사진과 기존 큐브 사진에 돌려 비교한다. 정답 라벨이 없으므로 개수, 모델 간 일치, 칸 시트만 낸다. 입력 사진과 이전 버전은 읽기만 한다.
# 사용: python 50_esp_compare.py
import collections as C, glob
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from infer import detect, merge, open_photo, starts, load_cls, classify, iou_ios

OUT = VER / 'runs/esp_compare'; OUT.mkdir(parents=True, exist_ok=True); RUNS = dict(v1=(V1 / 'runs/det/third_pseudo/weights/best.pt', 0.65), v2=(FLOWER / 'v2_seg/runs/seg/first/weights/best.pt', 0.65), v3=(VER / 'runs/seg/A_base/weights/best.pt', 0.60))
NEW = sorted(glob.glob(str(FLOWER.parent / 'model_git/v10_farm_infer/esp_cam_20261103/inputs/*.jfif'))); OLD = sorted(p for p in glob.glob(str(V1 / 'cube/**/*'), recursive=True) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')))
SETS = dict(new_esp=NEW, old_cube=OLD); print({k: len(v) for k, v in SETS.items()}, flush=True); net = load_cls(V1 / 'runs/cls/effnetv2s_r2/best.pt')


def det_box(model, im, tile=1280, overlap=0.25, imgsz=640, batch=32):
    views = []; side = max(im.size); P = []
    for y in starts(im.height, tile, overlap):
        for x in starts(im.width, tile, overlap):
            c = Image.new('RGB', (tile, tile), (114, 114, 114)); c.paste(im.crop((x, y, min(x + tile, im.width), min(y + tile, im.height))), (0, 0)); views.append((c.resize((imgsz, imgsz), Image.BILINEAR), x, y, tile / imgsz))
    c = Image.new('RGB', (side, side), (114, 114, 114)); c.paste(im, (0, 0)); views.append((c.resize((imgsz, imgsz), Image.BILINEAR), 0, 0, side / imgsz)); n = 0
    for i in range(0, len(views), batch):
        vs = views[i:i + batch]
        for v, r in zip(vs, model.predict([t[0] for t in vs], imgsz=imgsz, conf=0.05, iou=0.6, verbose=False)):
            if not len(r.boxes):
                continue
            t = r.boxes.xyxy.cpu().numpy(); b = t * v[3]; b[:, [0, 2]] += v[1]; b[:, [1, 3]] += v[2]; full = v[3] != tile / imgsz; m = 2
            frag = np.zeros(len(b), bool) if full else ((t[:, 0] <= m) & (v[1] > 0)) | ((t[:, 1] <= m) & (v[2] > 0)) | ((t[:, 2] >= imgsz - m) & (v[1] + tile < im.width)) | ((t[:, 3] >= imgsz - m) & (v[2] + tile < im.height))
            P.append(np.c_[b, r.boxes.conf.cpu().numpy(), frag, np.arange(n, n + len(b))]); n += len(b)
    return merge(np.vstack(P))[:, :5] if P else np.zeros((0, 5))


models = {k: YOLO(str(w)) for k, (w, _) in RUNS.items()}; res = {}; S_ = ['봉오리', '개화중', '만개']
for sname, files in SETS.items():
    res[sname] = {}
    for f in files:
        im = open_photo(f); row = dict(size=im.size)
        for k, m in models.items():
            P = det_box(m, im) if k == 'v1' else detect(m, im)[0]; thr = RUNS[k][1]; Q = P[P[:, 4] >= thr]; pr = classify(net, im, Q[:, :4]) if len(Q) else np.zeros((0, 3)); row[k] = dict(all=[[round(float(x), 1) for x in p[:4]] + [round(float(p[4]), 3)] for p in P if p[4] >= 0.2], stage=[S_[int(i)] for i in pr.argmax(1)] if len(pr) else [], n=len(Q), n05=int((P[:, 4] >= 0.5).sum()), n03=int((P[:, 4] >= 0.3).sum()))
        res[sname][Path(f).name] = row; print(sname, Path(f).name[:18], im.size, {k: (row[k]['n'], row[k]['n05'], row[k]['n03']) for k in models}, flush=True)
(OUT / 'preds.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
for sname, rows in res.items():
    print('==', sname, len(rows)); 
    for k in models:
        n = [r[k]['n'] for r in rows.values()]; sz = [max(p[2] - p[0], p[3] - p[1]) for r in rows.values() for p in r[k]['all'] if p[4] >= RUNS[k][1]]; st = C.Counter(x for r in rows.values() for x in r[k]['stage']); print(k, '임계값', RUNS[k][1], '합계', sum(n), '있는 사진', sum(x > 0 for x in n), '0.5', sum(r[k]['n05'] for r in rows.values()), '0.3', sum(r[k]['n03'] for r in rows.values()), st.most_common(), '긴 변 중앙값', round(float(np.median(sz)), 0) if sz else '-', flush=True)
    both = C.Counter()
    for r in rows.values():
        a = {k: np.array([p for p in r[k]['all'] if p[4] >= RUNS[k][1]]).reshape(-1, 5) for k in models}
        for x, y in (('v3', 'v1'), ('v3', 'v2')):
            both[x + '&' + y] += sum(bool(len(a[y])) and iou_ios(p[:4], a[y][:, :4])[0].max() >= 0.5 for p in a[x]); both[x + ' 전체'] = both[x + ' 전체'] + (len(a[x]) if y == 'v1' else 0)
    print(dict(both), flush=True)
# 칸 시트: v3 임계값 0.3 이상 예측을 신뢰도 순으로, 다른 모델의 판정과 함께
font = ImageFont.truetype(str(next(Path('/usr/share/fonts').rglob('*.ttf'))), 13) if list(Path('/usr/share/fonts').rglob('*.ttf')) else ImageFont.load_default()
path = {Path(f).name: f for fs in SETS.values() for f in fs}
for sname, rows in res.items():
    items = [(n, p) for n, r in rows.items() for p in r['v3']['all'] if p[4] >= 0.3]; items.sort(key=lambda t: -t[1][4]); items = items[:70]; W = 150; H = W + 18; S = Image.new('RGB', (10 * W, ((len(items) + 9) // 10) * H or H), (25, 25, 25)); d = ImageDraw.Draw(S); order = sorted(range(len(items)), key=lambda i: items[i][0]); cur = None
    for i in order:
        n, p = items[i]
        if cur != n:
            im = open_photo(path[n]); cur = n
        s = max(p[2] - p[0], p[3] - p[1]) * 1.8 + 24; cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2; x0, y0 = cx - s / 2, cy - s / 2; k = W / s; t = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((W, W)); ImageDraw.Draw(t).rectangle([(p[0] - x0) * k, (p[1] - y0) * k, (p[2] - x0) * k, (p[3] - y0) * k], outline=(0, 120, 255), width=2)
        x, y = (i % 10) * W, (i // 10) * H; S.paste(t, (x, y + 18)); d.text((x + 3, y + 2), f'{i + 1} v3 {p[4]:.2f} L{max(p[2] - p[0], p[3] - p[1]):.0f}', fill=(255, 255, 255), font=font)
    S.save(OUT / f'cells_{sname}.jpg', quality=85); (OUT / f'cells_{sname}.json').write_text(json.dumps([[n, p] for n, p in items]), encoding='utf-8'); print('cells', sname, len(items), flush=True)
# 전체 사진 그림: 새 ESP 1장과 큐브 2장
for sname, names in (('new_esp', list(res['new_esp'])[:2]), ('old_cube', sorted(res['old_cube'], key=lambda n: -res['old_cube'][n]['v3']['n'])[:2])):
    for n in names:
        im = open_photo(path[n]); z = 900 / im.width; im = im.resize((900, round(im.height * z))); d = ImageDraw.Draw(im); r = res[sname][n]
        for k, c in (('v1', (0, 200, 255)), ('v2', (255, 230, 0)), ('v3', (255, 60, 160))):
            for p in r[k]['all']:
                if p[4] >= RUNS[k][1]:
                    d.rectangle([p[0] * z, p[1] * z, p[2] * z, p[3] * z], outline=c, width=2)
        im.save(OUT / f'photo_{sname}_{n[:8]}.jpg', quality=82)
print('ESP_DONE', flush=True)
