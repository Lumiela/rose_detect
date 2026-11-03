# 오탐 시트의 사람 판정을 라벨 원본에 반영한다. 번호와 박스는 errors csv를 따른다. 한 번만 실행한다.
# 잎, 줄기=변경 없음 / 단계 판정=라벨 추가(같은 꽃의 라벨이 있으면 박스 유지) / 알수없음=제외 영역(같은 꽃의 라벨은 삭제)
# 그 외 일부만 잡힌 판정=같은 꽃의 라벨이 있으면 변경 없음, 없으면 제외 영역(라벨 없는 꽃이 배경으로 학습되지 않게)
# 미탐 시트(45)를 먼저 반영한 뒤 실행한다.
# 사용: python 46_apply_review_fp.py <판정 json> <errors csv>
import json, sys
from common import *

J = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); E = {r['id']: r for r in read(sys.argv[2])}
ann, ign = read(ANN), read(IGN); n = dict(added=0, same=0, ignored=0, dropped=0, none=0)
NOOP, UNK = {'잎', '줄기'}, {'알수없음', '알 수 없음', '제외'}


def same_label(b, f, ios_min=0.6):
    best, bi = None, 0
    for r in ann:
        if r['photo'] != f:
            continue
        g = box(r); w = max(0, min(b[2], g[2]) - max(b[0], g[0])); h = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = w * h
        ab, ag = (b[2] - b[0]) * (b[3] - b[1]), (g[2] - g[0]) * (g[3] - g[1]); iou, ios = i / (ab + ag - i + 1e-9), i / min(ab, ag)
        if (iou >= 0.3 or ios >= ios_min) and iou + ios > bi:
            best, bi = r, iou + ios
    return best


for k, j in J.items():
    e = E[k]; assert e['kind'] == 'fp', k; f = e['photo']
    assert not any(r['photo'] == f and r['ann_id'] == 'R' + k[1:] for r in ann), '이미 반영됨'
    if j in NOOP:
        n['none'] += 1; continue
    g = same_label(box(e), f, 0.6 if j in STAGES or j in UNK else 0.3)  # 일부 판정은 잘린 박스라 겹침이 작아도 같은 꽃으로 본다
    if j in STAGES:
        if g is not None:
            g['stage'] = j; n['same'] += 1
        else:
            ann.append(dict(photo=f, ann_id='R' + k[1:], x0=e['x0'], y0=e['y0'], x1=e['x1'], y1=e['y1'], stage=j, source='review')); n['added'] += 1
    elif j in UNK or g is None:
        if g is not None:
            ann.remove(g); n['dropped'] += 1
        ign.append({c: e[c] for c in IGN_FIELDS}); n['ignored'] += 1
    else:
        n['same'] += 1
# 방금 추가한 라벨과 겹치는 제외 영역은 지운다. 미탐 시트에서 일부로 판정해 옮긴 같은 꽃의 박스다
new = [r for r in ann if r['source'] == 'review' and 'R' + r['ann_id'][1:] == r['ann_id'] and 'F' + r['ann_id'][1:] in J and r['photo'] == E['F' + r['ann_id'][1:]]['photo']]


def covered(g):
    for r in new:
        if r['photo'] != g['photo']:
            continue
        a, c = box(g), box(r); w = max(0, min(a[2], c[2]) - max(a[0], c[0])); h = max(0, min(a[3], c[3]) - max(a[1], c[1]))
        if w * h / min((a[2] - a[0]) * (a[3] - a[1]), (c[2] - c[0]) * (c[3] - c[1])) >= 0.6:
            return True
    return False


before = len(ign); ign = [g for g in ign if not covered(g)]; n['ignore_released'] = before - len(ign)
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
