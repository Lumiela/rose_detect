# 오탐 시트의 사람 판정을 라벨 원본에 반영한다. 번호와 박스는 errors csv를 따른다. 한 번만 실행한다.
# 잎, 줄기=변경 없음 / 단계 판정=라벨 추가(같은 꽃의 라벨이 있으면 박스 유지) / 알수없음=제외 영역(같은 꽃의 라벨은 삭제)
# 그 외 일부만 잡힌 판정=같은 꽃의 라벨이 있으면 변경 없음, 없으면 제외 영역(라벨 없는 꽃이 배경으로 학습되지 않게)
# 사용: python 46_apply_review_fp.py <판정 json> <errors csv>
import json, sys
from common import *

J = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); E = {r['id']: r for r in read(sys.argv[2])}
ann, ign = read(ANN), read(IGN); n = dict(added=0, same=0, ignored=0, dropped=0, none=0)
NOOP, UNK = {'잎', '줄기'}, {'알수없음', '알 수 없음', '제외'}


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


for k, j in J.items():
    e = E[k]; assert e['kind'] == 'fp', k; f = e['photo']
    assert not any(r['photo'] == f and r['ann_id'] == 'R' + k[1:] for r in ann), '이미 반영됨'
    if j in NOOP:
        n['none'] += 1; continue
    g = same_label(box(e), f)
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
ann.sort(key=lambda r: (r['photo'], r['ann_id']))
write(ANN, ANN_FIELDS, ann); write(IGN, IGN_FIELDS, ign); print(n, 'annotations', len(ann), 'ignore', len(ign))
