# v1 라벨 원본에서 지운 중복 박스의 마스크를 뺀다. 나머지 줄은 그대로 둔다. 한 번만 실행한다.
# 사용: python 13_drop_masks.py
from common import *

drop = {(r['photo'], r['removed']) for r in read(V1 / 'labels' / 'review_duplicates_r1.csv')}; lines = MASKS.read_text(encoding='utf-8').splitlines(); keep = [t for t in lines if (lambda m: (m['photo'], m['key']) not in drop)(json.loads(t))]
MASKS.write_text('\n'.join(keep) + '\n', encoding='utf-8'); print('masks', len(lines), '->', len(keep))
