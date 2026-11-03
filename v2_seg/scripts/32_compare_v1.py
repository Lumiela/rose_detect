# v1 third_pseudo와 v2 세그멘테이션의 test 결과를 사진별로 비교하고, 예측 다각형을 사진에 그린다. 평가 파일은 고치지 않는다.
# 초록 맞게 찾음, 빨강 오탐, 파랑 놓침.
# 사용: python 32_compare_v1.py <v2 평가 폴더> <출력 폴더> [--photos 134905,135052]
import argparse, collections as C
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('v2'); ap.add_argument('out'); ap.add_argument('--photos', default='134905,135052,140440'); ap.add_argument('--width', type=int, default=800); a = ap.parse_args()
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
V1 = V1 / 'runs/det/third_pseudo/eval'; V2 = Path(a.v2); M1, M2 = J(V1 / 'photo_metrics.json'), J(V2 / 'photo_metrics.json'); P1, P2 = J(V1 / 'preds_test.json'), J(V2 / 'preds_test.json')
c1, c2 = M1['val']['chosen']['conf'], M2['val']['chosen']['conf']; anns, igns = C.defaultdict(list), C.defaultdict(list); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))


def match(f, P):
    # (맞은 예측, 오탐, 놓친 라벨)
    G = [box(r) for r in anns[f]]; used = set(); tp, fp = [], []
    for p in sorted(P, key=lambda q: -q[4]):
        v = [ov(p, g)[0] if j not in used else 0 for j, g in enumerate(G)]; j = max(range(len(v)), key=v.__getitem__) if v else -1
        if j >= 0 and v[j] >= 0.5:
            used.add(j); tp.append((p, j))
        elif not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in igns[f]):
            fp.append(p)
    return tp, fp, [j for j in range(len(G)) if j not in used]


rows = []; pair = C.Counter(); stage = C.defaultdict(C.Counter)
for f in sorted(P1):
    a1, a2 = [p for p in P1[f] if p[4] >= c1], [p for p in P2[f] if p[4] >= c2]; t1, f1, m1 = match(f, a1); t2, f2, m2 = match(f, a2); n = len(anns[f])
    if n or f1 or f2:
        rows.append([f[9:15], n, len(t1), len(t2), len(f1), len(f2), len(m1), len(m2)])
    s1, s2 = set(m1), set(m2); pair['both_miss'] += len(s1 & s2); pair['only_v1_miss'] += len(s1 - s2); pair['only_v2_miss'] += len(s2 - s1)
    for j in s1 - s2:
        stage['fixed'][anns[f][j]['stage']] += 1
    for j in s2 - s1:
        stage['new_miss'][anns[f][j]['stage']] += 1
    pair['fp_v1_also_near_v2'] += sum(any(ov(p, q)[0] >= 0.3 for q in a2) for p in f1); pair['fp_v1_total'] += len(f1); pair['fp_v2_also_near_v1'] += sum(any(ov(p, q)[0] >= 0.3 for q in a1) for p in f2); pair['fp_v2_total'] += len(f2)
    if any(k in f for k in a.photos.split(',')):
        im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); z = a.width / im.width; im = im.resize((a.width, round(im.height * z)), Image.BILINEAR); d = ImageDraw.Draw(im)
        def poly(p, c):
            if p[5]:
                q = [(x * z, y * z) for x, y in p[5]]; d.line(q + q[:1], fill=c, width=2)
        for p, _ in t2:
            poly(p, (40, 230, 80))
        for p in f2:
            poly(p, (255, 40, 40))
        for j in m2:
            g = box(anns[f][j]); d.rectangle([g[0] * z - 2, g[1] * z - 2, g[2] * z + 2, g[3] * z + 2], outline=(60, 140, 255), width=2)
        im.save(out / f'v2_{f[9:15]}.jpg', quality=84)
write(out / 'per_photo.csv', ['photo', 'label', 'tp_v1', 'tp_v2', 'fp_v1', 'fp_v2', 'miss_v1', 'miss_v2'], [dict(zip(['photo', 'label', 'tp_v1', 'tp_v2', 'fp_v1', 'fp_v2', 'miss_v1', 'miss_v2'], r)) for r in rows])
(out / 'compare.json').write_text(json.dumps(dict(pair=pair, stage={k: dict(v) for k, v in stage.items()}, rows=rows), ensure_ascii=False, indent=1), encoding='utf-8')
print(dict(pair), {k: dict(v) for k, v in stage.items()})
for r in rows:
    print(r)
