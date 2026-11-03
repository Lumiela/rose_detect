# 오류 시트의 사람 판정을 라벨 원본에 반영한다. 한 번만 실행한다.
# 봉오리 판정은 추가하되, 기존 라벨과 같은 꽃(SAME)은 추가하지 않고 기존 박스를 유지한다.
# 사용: python 44_apply_review.py <판정 json> <errors csv>
import json, sys
from common import *

J = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); E = {r['id']: r for r in read(sys.argv[2])}
SAME = {'F003', 'F012', 'F013', 'F022', 'F054', 'F057', 'F070', 'F071', 'F073', 'F080', 'F087', 'F088'}  # 박스 크기만 다른 같은 꽃
IGNORE = {'F029', 'F062', 'F065', 'F068', 'F039', 'F081', 'F091', 'F006'}  # 제외, 알수없음, 사진 가장자리에서 잘린 봉오리, 단계 미정(F006)
DROP = {('20261030_135052.jpg', 'D037')}  # F062 알수없음과 같은 대상
STAGE = {('20261030_134908.jpg', 'D019'): '봉오리'}  # F013 판정
ann, ign = read(ANN), read(IGN)
assert not any(r['source'] == 'review' for r in ann), '이미 반영됨'
assert all(k in {(r['photo'], r['ann_id']) for r in ann} for k in DROP | set(STAGE))
ann = [r for r in ann if (r['photo'], r['ann_id']) not in DROP]
for r in ann:
    r['stage'] = STAGE.get((r['photo'], r['ann_id']), r['stage'])
n = dict(added=0, ignored=0)
for k, j in J.items():
    e = E[k]
    if k in IGNORE:
        ign.append({c: e[c] for c in IGN_FIELDS}); n['ignored'] += 1
    elif j == '봉오리' and k not in SAME:
        ann.append(dict(photo=e['photo'], ann_id='R' + k[1:], x0=e['x0'], y0=e['y0'], x1=e['x1'], y1=e['y1'], stage='봉오리', source='review')); n['added'] += 1
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
