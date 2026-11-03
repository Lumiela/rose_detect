# 40_flower_ids.py 결과가 31_eval_seg.py와 v1 51_eval_e2e.py 결과와 같은지 확인한다. ID를 붙이는 동안 꽃이나 단계가 바뀌지 않아야 한다.
# ID 하나에 꽃 하나가 맞는지도 라벨로 센다. 합쳐짐은 한 마스크 안에 라벨 중심이 2개 이상, 쪼개짐은 한 라벨 박스 안에 들어가는 ID가 2개 이상인 경우다.
# 사용: python 41_check_ids.py <flowers.json> <평가 폴더>   (결과는 flowers.json 옆의 check.json)
import collections as C, sys
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import cv2
import numpy as np

D = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); E = Path(sys.argv[2]); e2e = json.loads((E / 'e2e_metrics.json').read_text(encoding='utf-8')); conf = e2e['conf']; IOU = 0.5
man = {m['photo']: m['split'] for m in read(MANIFEST)}; anns, igns = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    anns[r['photo']].append(r)
for r in read(IGN):
    igns[r['photo']].append(box(r))
probs = C.defaultdict(list); merged = C.defaultdict(list); split = C.defaultdict(list)
bad = C.Counter(); cm = {s: np.zeros((4, 4), int) for s in e2e if s in ('val', 'test')}; cnt = {}; preds = {s: json.loads((E / f'preds_{s}.json').read_text(encoding='utf-8')) for s in cm}; overlap = []
for ph in D:
    f, F = ph['photo'], ph['flowers']; s = man[f]; P = [p for p in preds[s][f] if p[4] >= conf]
    if [x['idx'] for x in F] != list(range(1, len(F) + 1)) or len({x['id'] for x in F}) != len(F):
        bad['번호'] += 1
    a_ = sorted((tuple(p[:4]), p[5]) for p in P); b_ = sorted(((x['x0'], x['y0'], x['x1'], x['y1']), x['poly']) for x in F)
    if a_ != b_:
        bad['박스나 다각형'] += 1
    gt = anns[f]; used = set(); it = []
    order = {tuple(p[:4]): i for i, p in enumerate(P)}
    for x in sorted(F, key=lambda x: order.get((x['x0'], x['y0'], x['x1'], x['y1']), 0)):  # 평가와 같은 순서로 맞춘다
        b = [x['x0'], x['y0'], x['x1'], x['y1']]; v = [ov(b, box(r))[0] if j not in used else 0 for j, r in enumerate(gt)]; j = int(np.argmax(v)) if v else -1
        if j >= 0 and v[j] >= IOU:
            used.add(j); it.append((STAGES.index(gt[j]['stage']), x['stage_idx'])); probs[(x['stage_idx'], STAGES.index(gt[j]['stage']) == x['stage_idx'])].append(x['stage_prob']); continue
        if not any(q[0] <= x['cx'] <= q[2] and q[1] <= x['cy'] <= q[3] for q in igns[f]):
            it.append((3, x['stage_idx']))
    it += [(STAGES.index(r['stage']), 3) for j, r in enumerate(gt) if j not in used]
    for t, q in it:
        cm[s][t, q] += 1
    cnt[f] = [sum(y == i for _, y in it) for i in range(3)]
    H = np.zeros((ph['height'], ph['width']), np.uint16)  # 마스크끼리 겹치는 정도. 겹치는 ID 쌍을 센다
    for x in F:
        if len(x['poly']) < 3:
            continue
        m = np.zeros(H.shape, np.uint8); cv2.fillPoly(m, [np.array(x['poly'], np.int32)], 1); y = H[m > 0]; y = y[y > 0]
        if len(y):
            w, c = np.unique(y, return_counts=True)
            for o, k in zip(w, c):
                overlap.append(round(float(k / max(int(m.sum()), 1)), 3))
        H[m > 0] = x['idx']
    inside = C.defaultdict(list)
    for j, r in enumerate(gt):
        g = box(r); o_ = H[min(int((g[1] + g[3]) / 2), H.shape[0] - 1), min(int((g[0] + g[2]) / 2), H.shape[1] - 1)]
        if o_:
            inside[int(o_)].append(j)
        ids = [x['idx'] for x in F if g[0] <= x['cx'] <= g[2] and g[1] <= x['cy'] <= g[3] and ov([x['x0'], x['y0'], x['x1'], x['y1']], g)[1] >= 0.7]
        if len(ids) >= 2:
            split[s].append([f, j, ids])
    merged[s] += [[f, k, v] for k, v in inside.items() if len(v) >= 2]
res = {}
for s in cm:
    same = cm[s].tolist() == e2e[s]['confusion']; c2 = all(cnt[r['photo']] == r['pred'] for r in e2e[s]['per_photo'])
    print(s, '혼동 행렬 일치', same, '사진별 단계 개수 일치', c2)
    res[s] = dict(confusion_same=same, counts_same=c2, flowers=sum(len(p['flowers']) for p in D if man[p['photo']] == s), merged=merged[s], split=split[s])
    if not same:
        print(' 지금', cm[s].tolist(), '\n 평가', e2e[s]['confusion'])
area = [x['area'] for p in D for x in p['flowers']]; sol = np.array([x['solidity'] for p in D for x in p['flowers']]); o = np.array(overlap or [0.0])
print('사진', len(D), '꽃', len(area), '오류', dict(bad)); print('넓이 0', sum(v == 0 for v in area), '볼록도<0.8', int((sol < 0.8).sum()), '겹치는 쌍', len(overlap), '겹침>=0.5', int((o >= 0.5).sum())); q3 = lambda v: [len(v)] + np.percentile(v, [25, 50, 75]).round(3).tolist() if v else [0, 0, 0, 0]
res.update(photos=len(D), flowers=len(area), errors=dict(bad), area_zero=sum(v == 0 for v in area), low_solidity=int((sol < 0.8).sum()), overlap_pairs=len(overlap), overlap_ge50=int((o >= 0.5).sum()), stage_prob={STAGES[i]: dict(correct=q3(probs[(i, True)]), wrong=q3(probs[(i, False)])) for i in range(3)})
(Path(sys.argv[1]).parent / 'check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for s in cm:
    print(s, '합쳐짐', len(res[s]['merged']), '쪼개짐', len(res[s]['split']))
print('CHECK_DONE', flush=True)
