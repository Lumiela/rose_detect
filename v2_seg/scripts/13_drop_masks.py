# v1 라벨 원본에서 지운 박스의 마스크를 빼고, 바뀐 단계를 맞춘다. 임시 라벨 줄과 나머지 줄은 그대로 둔다. 새로 추가된 라벨의 마스크는 만들지 않는다.
# 사용: python 13_drop_masks.py
from common import *

ann = {(r['photo'], r['ann_id']): r for r in read(ANN)}; lines = MASKS.read_text(encoding='utf-8').splitlines(); keep = []; n = 0
for t in lines:
    m = json.loads(t); r = ann.get((m['photo'], m['key']))
    if m['source'] != 'pseudo' and r is None:
        continue
    if m['source'] != 'pseudo' and m['stage'] != r['stage']:
        m['stage'] = r['stage']; t = json.dumps(m, ensure_ascii=False); n += 1
    keep.append(t)
MASKS.write_text('\n'.join(keep) + '\n', encoding='utf-8'); print('masks', len(lines), '->', len(keep), 'stage', n)
