# v2만 놓친 미판정 라벨 시트의 사람 판정을 라벨 원본에 반영한다. 번호와 박스는 labels/review_test_errors_r4.csv를 따른다. 한 번만 실행한다.
# 단계 판정=라벨 유지(단계가 다르면 수정), 일부=라벨을 제외 영역으로 옮김. 
# 사용: python 56_apply_review_u.py
import json
from common import *

L = VER / 'labels'; U = json.loads((L / 'review_test_u_r4.json').read_text(encoding='utf-8')); E = {r['id']: r for r in read(L / 'review_test_errors_r4.csv')}; assert set(U) == set(E) and len(U) == 35
ann, ign = read(ANN), read(IGN); idx = {(r['photo'], r['ann_id']): r for r in ann}; n = dict(kept=0, stage=0, ignored=0)
for k, j in U.items():
    r = idx[(E[k]['photo'], E[k]['ann_id'])]; print(k, r['photo'][9:15], r['ann_id'], r['stage'], j, r['x0'], r['y0'], r['x1'], r['y1'], flush=True) if j != '봉오리' or r['stage'] != '봉오리' else None
    if j in STAGES:
        if r['stage'] != j:
            r['stage'] = j; n['stage'] += 1
        else:
            n['kept'] += 1
    else:
        assert j in ('봉오리 일부', '꽃 일부'), k; ann.remove(r); ign.append({c: r[c] for c in IGN_FIELDS}); n['ignored'] += 1
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
