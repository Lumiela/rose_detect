# v3와 v1, v2를 같은 test 사진과 라벨로 비교한다. 임계값은 모델마다 val F1로 정한 값이다. 차이 구간은 test 사진을 복원 추출한 95% 구간이다. 결과: runs/seg/<이름>/eval/compare_v1_v2.json
# 사용: python 33_compare.py A_base
import collections as C, sys
from common import *
import numpy as np

assert_eval_same(); NAME = sys.argv[1]; J = lambda p: json.loads(Path(p).read_text(encoding='utf-8')); RS = {'v1': V1 / 'runs/det/third_pseudo', 'v2': FLOWER / 'v2_seg/runs/seg/first', 'v3': VER / 'runs/seg' / NAME}
M = {k: J(r / 'eval/photo_metrics.json') for k, r in RS.items()}; E = {k: J(r / 'eval/e2e_metrics.json') for k, r in RS.items()}; I3 = {k: J(r / 'eval_iou03/photo_metrics.json') for k, r in RS.items()}; P = {k: J(r / 'eval/preds_test.json') for k, r in RS.items()}; CONF = {k: M[k]['val']['chosen']['conf'] for k in RS}
PH = sorted(P['v3']); assert all(set(P[k]) == set(PH) for k in P); GT, IG = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    GT[r['photo']].append(r)
for r in read(IGN):
    IG[r['photo']].append(box(r))


def match(k, f, iou=0.5, neutral=False):
    rows = [r for r in GT[f] if not (neutral and r['ann_id'][0] in 'TV')]; G = [box(r) for r in rows]; I_ = IG[f] + ([box(r) for r in GT[f] if r['ann_id'][0] in 'TV'] if neutral else []); used = set(); tp = fp = 0
    for p in sorted((q for q in P[k][f] if q[4] >= CONF[k]), key=lambda q: -q[4]):
        v = [ov(p[:4], g)[0] if j not in used else 0 for j, g in enumerate(G)]; j = max(range(len(v)), key=v.__getitem__) if v else -1
        if j >= 0 and v[j] >= iou:
            used.add(j); tp += 1
        elif not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in I_):
            fp += 1
    return tp, fp, len(rows) - len(used), [rows[j] for j in range(len(rows)) if j not in used]


def prf(c):
    tp, fp, fn = c.sum(0); p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1); return np.array([p, r, 2 * p * r / max(p + r, 1e-9)])


T = lambda k, iou=0.5, n=False: np.array([match(k, f, iou, n)[:3] for f in PH], float)
cnt = {k: np.array([[abs(x - y) for x, y in zip(c['label'], c['pred'])] for c in sorted(E[k]['test']['per_photo'], key=lambda c: c['photo'])], float) for k in RS}
out = dict(conf=CONF, rows={})
print('## test 지표')
for k in RS:
    t = M[k]['test']['chosen']; e = E[k]['test']; print(k, 'conf', CONF[k], 'P', t['precision'], 'R', t['recall'], 'F1', t['f1'], 'fp', t['fp'], 'fn', t['fn'], 'F1@.3', I3[k]['test']['chosen']['f1'], 'macro', e['macro_f1'], e['macro_f1_photo_ci95'], 'count', e['count_mae_per_photo'], 'pred', e['count_total_pred'], 'label', e['count_total_label'], 'stageR', e['recall'], 'stageP', e['precision'])
    print('   recall_by', {x: t['recall_by'][x] for x in ('<80px', '80-160px', '>=160px', '봉오리', '개화중', '만개', 'test_red', 'test_yellow', 'test_mixed')}); print('   추가 라벨 제외 P,R,F1', [round(float(x), 4) for x in prf(T(k, 0.5, True))], 'IoU .3', [round(float(x), 4) for x in prf(T(k, 0.3, True))])
print('## v3-기준 95% 구간')
for b in ('v1', 'v2'):
    g = np.random.default_rng(0); N = len(PH); rows = {}
    for name, A, B in (('IoU.5', T('v3'), T(b)), ('IoU.3', T('v3', 0.3), T(b, 0.3)), ('제외 IoU.5', T('v3', 0.5, True), T(b, 0.5, True)), ('제외 IoU.3', T('v3', 0.3, True), T(b, 0.3, True))):
        D = []
        for _ in range(4000):
            k = g.integers(N, size=N); D.append(prf(A[k]) - prf(B[k]))
        lo, hi = np.percentile(D, [2.5, 97.5], axis=0); rows[name] = dict(P=[round(float(lo[0]), 3), round(float(hi[0]), 3)], R=[round(float(lo[1]), 3), round(float(hi[1]), 3)], F1=[round(float(lo[2]), 3), round(float(hi[2]), 3)])
    D = [cnt['v3'][k].mean(0) - cnt[b][k].mean(0) for k in (g.integers(N, size=N) for _ in range(4000))]; lo, hi = np.percentile(D, [2.5, 97.5], axis=0); rows['개수 오차'] = dict(lo=[round(float(x), 2) for x in lo], hi=[round(float(x), 2) for x in hi]); rows['봉오리 개수가 가까운 사진'] = [int((cnt['v3'][:, 0] < cnt[b][:, 0]).sum()), int((cnt['v3'][:, 0] > cnt[b][:, 0]).sum()), int((cnt['v3'][:, 0] == cnt[b][:, 0]).sum())]
    out['rows'][b] = rows; print('v3 -', b, json.dumps(rows, ensure_ascii=False))
miss = {k: {f: {r['ann_id'] for r in match(k, f)[3]} for f in PH} for k in RS}; both = sum(len(miss['v3'][f] & miss['v2'][f]) for f in PH); only3 = sum(len(miss['v3'][f] - miss['v2'][f]) for f in PH); only2 = sum(len(miss['v2'][f] - miss['v3'][f]) for f in PH); print('놓침 v3와 v2: 둘 다', both, 'v3만', only3, 'v2만', only2); o1 = sum(len(miss['v3'][f] - miss['v1'][f]) for f in PH); o2 = sum(len(miss['v1'][f] - miss['v3'][f]) for f in PH); print('놓침 v3와 v1: 둘 다', sum(len(miss['v3'][f] & miss['v1'][f]) for f in PH), 'v3만', o1, 'v1만', o2)
print('사진별 놓침', [(f[9:15], len(GT[f]), *[match(k, f)[2] for k in RS], *[match(k, f)[1] for k in RS]) for f in PH if GT[f]])
(RS['v3'] / 'eval/compare_v1_v2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8'); print('COMPARE_DONE')
