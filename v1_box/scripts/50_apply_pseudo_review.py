# 임시 라벨 표본의 사람 판정을 라벨 원본에 반영한다. 한 번만 실행한다.
# 단계 판정=라벨 추가 / 알수없음, 부분=제외 영역 / 잎, 줄기=pseudo_rejected.csv(다시 임시 라벨이 되지 않게 막는다)
# 사용: python 50_apply_pseudo_review.py <판정 json> <표본 csv>
import json, sys
from common import *

J = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); E = {r['id']: r for r in read(sys.argv[2])}; assert set(J) == set(E)
ann, ign, rej = read(ANN), read(IGN), read(REJ); n = dict(added=0, ignored=0, rejected=0)
assert not any(r['ann_id'].startswith('P') for r in ann), '이미 반영됨'
for k, j in J.items():
    e = E[k]
    if j in STAGES:
        ann.append(dict(photo=e['photo'], ann_id=k, x0=e['x0'], y0=e['y0'], x1=e['x1'], y1=e['y1'], stage=j, source='review')); n['added'] += 1
    elif j in ('잎', '줄기'):
        rej.append({c: e[c] for c in IGN_FIELDS}); n['rejected'] += 1
    else:
        ign.append({c: e[c] for c in IGN_FIELDS}); n['ignored'] += 1
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); write(REJ, IGN_FIELDS, rej); print(n, 'annotations', len(ann), 'ignore', len(ign), 'rejected', len(rej))
