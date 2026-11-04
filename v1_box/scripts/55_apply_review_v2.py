# v2_seg test 오탐 시트의 사람 판정을 라벨 원본에 반영한다. 번호와 박스는 labels/review_test_errors_r3.csv를 따른다. 한 번만 실행한다.
# 잎=변경 없음, 봉오리와 꽃=같은 꽃의 라벨이 있으면 그 박스를 유지하고 봉오리에 라벨이 없으면 추가, 잘린 봉오리와 알수없음=제외 영역.
# 사용: python 55_apply_review_v2.py
import json
from common import *

L = VER / 'labels'; FP = json.loads((L / 'review_test_fp_r3.json').read_text(encoding='utf-8')); E = {r['id']: r for r in read(L / 'review_test_errors_r3.csv')}
ann, ign = read(ANN), read(IGN); n = dict(leaf=0, same=0, added=0, ignored=0); assert len(FP) == 18 and not any(r['ann_id'].startswith('V') for r in ann), '이미 반영됨'


def same_label(b, f):
    best, bi = None, 0
    for r in ann:
        if r['photo'] != f:
            continue
        g = box(r); w = max(0, min(b[2], g[2]) - max(b[0], g[0])); h = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = w * h
        ab, ag = (b[2] - b[0]) * (b[3] - b[1]), (g[2] - g[0]) * (g[3] - g[1]); iou, ios = i / (ab + ag - i + 1e-9), i / min(ab, ag)
        if (iou >= 0.3 or ios >= 0.6) and iou + ios > bi:
            best, bi = r, iou + ios
    return best


for k, j in FP.items():
    e = E[k]; f = e['photo']; assert e['kind'] == 'fp', k; g = same_label(box(e), f)
    print(k, f[9:15], j, [e[c] for c in ('x0', 'y0', 'x1', 'y1')], e['conf'], e['iou'], (g['ann_id'], g['stage']) if g else None, flush=True)
    if j == '잎':
        n['leaf'] += 1
    elif j in ('봉오리', '꽃', '잎에 가려진 꽃') and g is not None:
        n['same'] += 1
    elif j == '봉오리':
        ann.append(dict(photo=f, ann_id='V' + k[1:], x0=e['x0'], y0=e['y0'], x1=e['x1'], y1=e['y1'], stage=j, source='review')); n['added'] += 1
    else:
        assert g is None and j in ('알수없음', '흐려서 알수없음', '꽃봉오리 하단'), k; ign.append({c: e[c] for c in IGN_FIELDS}); n['ignored'] += 1
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
