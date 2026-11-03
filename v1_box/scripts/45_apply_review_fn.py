# 미탐 시트의 사람 판정을 라벨 원본에 반영한다. 한 번만 실행한다.
# 잎=라벨 삭제, 알수없음과 일부만 보이는 대상=라벨을 제외 영역으로 옮김, 개화중=단계 수정. 나머지는 라벨 유지.
# 사용: python 45_apply_review_fn.py <판정 json> <errors csv>
import json, sys
from common import *

J = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); E = {r['id']: r for r in read(sys.argv[2])}
ann, ign = read(ANN), read(IGN); idx = {(r['photo'], r['ann_id']): r for r in ann}
PART = {'꽃 일부', '봉오리 일부', '봉오리 우측 일부', '봉오리 좌측 일부', '봉오리 우측과 잎'}
drop, n = set(), dict(leaf=0, ignored=0, stage=0, already_removed=0)
for k, j in J.items():
    key = (E[k]['photo'], E[k]['ann_id']); r = idx.get(key)
    if r is None:
        assert k == 'M096', k  # F062 판정 때 이미 지움
        n['already_removed'] += 1; continue
    if j == '잎':
        drop.add(key); n['leaf'] += 1
    elif j == '알수없음' or j in PART:
        drop.add(key); ign.append({c: r[c] for c in IGN_FIELDS}); n['ignored'] += 1
    elif j == '개화중' and r['stage'] != '개화중':
        r['stage'] = '개화중'; n['stage'] += 1
ann = [r for r in ann if (r['photo'], r['ann_id']) not in drop]
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
