# v1 라벨을 복사해 팀 교차 판정과 사용자 판단의 최종 처리 안, train 표본 확인 13칸을 반영한 v3 라벨을 만든다. v1_box/labels는 읽기만 한다. 기본은 미리 보기이고 --apply를 줘야 파일을 쓴다.
# 라벨 추가(ann_id=H....), 제외 영역, 차단 목록. 새 라벨끼리 겹치면 모델 신뢰도가 높은 쪽만 남긴다. 사람이 이미 제외하거나 차단한 곳은 덮지 않는다. 새 제외 영역이나 차단이 새 라벨과 겹치면 라벨을 남긴다.
# 사용: python 01_apply_team_check.py [--plan ../human_label/최종처리_안.csv] [--apply]
import argparse, collections as C, shutil
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--plan', default=str(FLOWER / 'human_label/최종처리_안.csv')); ap.add_argument('--apply', action='store_true'); a = ap.parse_args()
L1 = V1 / 'labels'; L3 = VER / 'labels'; man = {m['photo']: m for m in read(L1 / 'photos.csv')}; ann, ign, rej = read(L1 / 'annotations.csv'), read(L1 / 'ignore_regions.csv'), read(L1 / 'pseudo_rejected.csv'); plan = read(a.plan); assert len(plan) == 1713
assert not (L3 / 'annotations.csv').exists() or not a.apply, '이미 반영했습니다'
assert not any(r['ann_id'].startswith('H') for r in ann); n_ann, n_ign, n_rej = len(ann), len(ign), len(rej); split0 = C.Counter(man[r['photo']]['split'] for r in ann)
hit = lambda p, q: (lambda v: v[0] >= 0.3 or v[1] >= 0.6)(ov(p, q)); full = lambda s: f'20261030_{s}.jpg'
UC = json.loads((L1 / 'user_check_r1.json').read_text(encoding='utf-8')); idx = {(r['photo'], r['ann_id']): r for r in ann}; check = []
for r in read(L1 / 'user_check_r1.csv'):
    v = UC.get(r['id'], '맞음')
    if v == '맞음':
        continue
    k = (r['photo'], r['key']); had = k in idx; check.append(dict(id=r['id'], photo=r['photo'], key=r['key'], source=r['source'], verdict=v, action=('라벨 삭제, ' if had else '') + ('차단' if v == '아님' else '제외 영역')))
    if had:
        ann.remove(idx[k])
    (rej if v == '아님' else ign).append({c: r[c] for c in IGN_FIELDS})
assert len(check) == 13
adds, igs, rjs = C.defaultdict(list), C.defaultdict(list), C.defaultdict(list)
for r in plan:
    f = full(r['photo']); assert man[f]['split'] == 'train', f; b = dict(r, photo=f, x0=float(r['x0']), y0=float(r['y0']), x1=float(r['x1']), y1=float(r['y1']), stage=r['최종'], conf=float(r['conf']))
    {'라벨 추가': adds, '제외 영역': igs, '차단': rjs}[r['처리']][f].append(b)
keep = C.defaultdict(list); skipped = []
for f, xs in adds.items():
    for b in sorted(xs, key=lambda x: -x['conf']):
        if any(hit(box(b), box(k)) for k in keep[f]):
            skipped.append(dict(b, why='새 라벨끼리 겹침'))
        else:
            keep[f].append(b)
exist = C.defaultdict(list); [exist[r['photo']].append(box(r)) for r in ann + ign + rej]
for f in list(keep):
    good = []
    for k in keep[f]:
        if any(hit(box(k), g) for g in exist[f]):
            skipped.append(dict(k, why='기존 라벨, 제외 영역, 차단과 겹침'))
        else:
            good.append(k)
    keep[f] = good
nig, nrj = [], []
for src, dst in ((igs, nig), (rjs, nrj)):
    for f, xs in src.items():
        for b in xs:
            if any(hit(box(b), box(k)) for k in keep.get(f, [])):
                skipped.append(dict(b, why='새 라벨과 겹쳐 뺌'))
            else:
                dst.append(b)
new = []; added = []; i = 0
for f in sorted(keep):
    for b in sorted(keep[f], key=lambda x: (x['y0'], x['x0'])):
        i += 1; aid = f'H{i:04d}'; new.append(dict(photo=f, ann_id=aid, x0=b['x0'], y0=b['y0'], x1=b['x1'], y1=b['y1'], stage=b['stage'], source='team')); added.append(dict(ann_id=aid, photo=f, conf=b['conf'], 근거=b['근거'], 등급=b['등급'], 판정자1=b['판정자1'], 판정1=b['판정1'], 판정자2=b['판정자2'], 판정2=b['판정2'], Claude=b['Claude'], 사용자=b['사용자']))
ann2 = ann + new; ign2 = ign + [{c: b[c] for c in IGN_FIELDS} for b in nig]; rej2 = rej + [{c: b[c] for c in IGN_FIELDS} for b in nrj]
print('라벨', n_ann, '->', len(ann2), '추가', len(new), dict(C.Counter(r['stage'] for r in new)), '삭제', n_ann - len(ann)); print('분할별', dict(split0), '->', dict(C.Counter(man[r['photo']]['split'] for r in ann2))); print('제외 영역', n_ign, '->', len(ign2), '차단', n_rej, '->', len(rej2)); print('뺀 칸', dict(C.Counter(s['why'] for s in skipped)))
if a.apply:
    L3.mkdir(exist_ok=True); shutil.copy2(L1 / 'photos.csv', L3 / 'photos.csv'); write(L3 / 'annotations.csv', ANN_FIELDS, ann2); write(L3 / 'ignore_regions.csv', IGN_FIELDS, ign2); write(L3 / 'pseudo_rejected.csv', IGN_FIELDS, rej2)
    write(L3 / 'team_added.csv', list(added[0]), added); write(L3 / 'user_check_applied.csv', list(check[0]), check); write(L3 / 'team_skipped.csv', ['photo', 'x0', 'y0', 'x1', 'y1', '처리', 'why'], [{k: s[k] for k in ('photo', 'x0', 'y0', 'x1', 'y1', '처리', 'why')} for s in skipped]); print('APPLIED')
else:
    print('DRY_RUN')
