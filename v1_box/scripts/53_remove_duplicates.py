# 라벨 원본에서 한 봉오리에 박스가 2개 있는 중복을 지운다. 모두 Claude 초안에서 생긴 것이다. 한 번만 실행한다.
# 남길 박스는 SAM 마스크를 감싼 박스와 더 잘 맞는 쪽이다(모델 예측은 보지 않는다). 값이 같거나 마스크가 없으면 그림을 보고 정했다.
# 지운 내역은 labels/review_duplicates_r1.csv에 남긴다.
# 사용: python 53_remove_duplicates.py
from common import *

PAIRS = [('20261030_134043.jpg', 'D014', 'D040', '마스크 0.787, 0.790'), ('20261030_134043.jpg', 'D021', 'D015', '마스크 0.81, 0.83'), ('20261030_134043.jpg', 'D031', 'D020', '같은 박스'),
         ('20261030_134242.jpg', 'D031', 'D012', '마스크 0.75, 0.80'), ('20261030_134537.jpg', 'D029', 'D033', '마스크 없음(buffer), 그림 확인'), ('20261030_134601.jpg', 'D037', 'D039', '마스크 0.71, 0.77'),
         ('20261030_135007.jpg', 'D012', 'D028', '마스크 0.767, 0.769'), ('20261030_140419.jpg', 'D035', 'D022', '마스크 0.89, 0.93'), ('20261030_140426.jpg', 'D029', 'D040', '마스크 0.86, 0.91'), ('20261030_140433.jpg', 'D026', 'D034', '마스크 0.91, 0.93')]
ann = read(ANN); key = {(r['photo'], r['ann_id']): r for r in ann}; log = []
for f, drop, keep, why in PAIRS:
    a, b = key[(f, drop)], key[(f, keep)]; assert a['stage'] == b['stage'], (f, drop)
    ann.remove(a); log.append(dict(photo=f, removed=drop, kept=keep, x0=a['x0'], y0=a['y0'], x1=a['x1'], y1=a['y1'], stage=a['stage'], source=a['source'], reason=why))
write(ANN, ANN_FIELDS, ann); write(VER / 'labels' / 'review_duplicates_r1.csv', list(log[0]), log); print('removed', len(log), 'annotations', len(ann))
