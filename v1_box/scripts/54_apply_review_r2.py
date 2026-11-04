# third_pseudo test 오류 시트의 사람 판정을 라벨 원본에 반영한다. 번호와 박스는 labels/review_test_errors_r2.csv를 따른다. 한 번만 실행한다.
# 미탐: 잎=라벨 삭제, 알수없음과 꽃 일부=라벨을 제외 영역으로 옮김, 봉오리인데 라벨 단계가 다르면 수정, 나머지는 유지.
# 오탐: 잎=변경 없음, 봉오리=같은 꽃의 라벨이 있으면 그 박스를 유지하고 없으면 추가, 꽃=박스 하나에 꽃 둘(라벨 유지), 봉오리 하단과 알수없음=제외 영역.
# 사용: python 54_apply_review_r2.py
import json
from common import *

L = VER / 'labels'; FP, FN = (json.loads((L / f'review_test_{k}_r2.json').read_text(encoding='utf-8')) for k in ('fp', 'fn')); E = {r['id']: r for r in read(L / 'review_test_errors_r2.csv')}
ann, ign = read(ANN), read(IGN); idx = {(r['photo'], r['ann_id']): r for r in ann}; n = dict(leaf=0, ignored=0, stage=0, kept=0, added=0, same=0, fp_leaf=0, merged=0, fp_ignored=0, ignore_released=0)
assert len(FP) == 86 and len(FN) == 81 and not any(r['ann_id'].startswith('T') for r in ann), '이미 반영됨'
for k, j in FN.items():
    r = idx[(E[k]['photo'], E[k]['ann_id'])]
    if j == '잎':
        ann.remove(r); n['leaf'] += 1
    elif j in ('알수없음', '꽃 일부'):
        ann.remove(r); ign.append({c: r[c] for c in IGN_FIELDS}); n['ignored'] += 1
    elif j == '봉오리' and r['stage'] != '봉오리':
        r['stage'] = '봉오리'; n['stage'] += 1
    else:
        n['kept'] += 1


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


new = []
for k, j in FP.items():
    e = E[k]; f = e['photo']; g = same_label(box(e), f)
    if j == '잎':
        n['fp_leaf'] += 1
    elif j == '봉오리':
        if g is None:
            r = dict(photo=f, ann_id='T' + k[1:], x0=e['x0'], y0=e['y0'], x1=e['x1'], y1=e['y1'], stage=j, source='review'); ann.append(r); new.append(r); n['added'] += 1
        else:
            assert g['stage'] == j, k; n['same'] += 1
    elif j == '꽃':
        assert g is not None, k; n['merged'] += 1
    else:
        assert g is None, k; ign.append({c: e[c] for c in IGN_FIELDS}); n['fp_ignored'] += 1


def covered(g):
    # 미탐 판정으로 제외 영역에 옮겼지만 오탐 판정에서 같은 꽃의 라벨이 새로 생긴 경우
    for r in new:
        if r['photo'] == g['photo']:
            a, c = box(g), box(r); w = max(0, min(a[2], c[2]) - max(a[0], c[0])); h = max(0, min(a[3], c[3]) - max(a[1], c[1]))
            if w * h / min((a[2] - a[0]) * (a[3] - a[1]), (c[2] - c[0]) * (c[3] - c[1])) >= 0.6:
                return True
    return False


m = len(ign); ign = [g for g in ign if not covered(g)]; n['ignore_released'] = m - len(ign)
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
