# 기존 labels.csv를 라벨 원본으로 옮기고 사진 목록과 분할을 만든다.
# 이미 annotations.csv가 있으면 라벨은 덮어쓰지 않고 분할만 다시 계산한다.
# 분할: 휴대폰 두 대가 같은 시각에 다른 재배대를 찍었으므로 시각 구간 단위로 나눈다.
import collections as C
from PIL import Image
from common import *

# (시작, 끝, 구간 이름, 분할). 시각은 HHMMSS, 양 끝 포함
WINDOWS = [
    ('134601', '134642', 'val_pink_red', 'val'),
    ('135007', '135052', 'test_yellow', 'test'),
    ('134905', '134938', 'test_mixed', 'test'),
    ('140419', '140440', 'test_red', 'test'),
    ('134849', '134849', 'buffer', 'buffer'),  # test_mixed 직전 S25U: 같은 재배대일 수 있음
    ('134856', '134856', 'buffer', 'buffer'),  # 134905와 같은 재배대로 보임. 어떤 학습에도 쓰지 않는다
    ('134529', '134550', 'buffer', 'buffer'),  # val 134601과 같은 재배대(초록 통, 분홍 꽃)가 보임
    ('140817', '140817', 'buffer', 'buffer'),  # test_red와 같은 재배대로 보임
]


def split_of(hms):
    for a, b, name, split in WINDOWS:
        if a <= hms <= b:
            return name, split
    return 'train_' + hms[:3], 'train'


photos = sorted(p.name for p in PHOTOS.glob('*.jpg'))
legacy = read(LEGACY_CSV)
if not ANN.exists():
    rows = [dict(photo=r['source_image'], ann_id=f"L{r['file'].split('_B')[-1].split('.')[0]}",
                 x0=r['obj_x0'], y0=r['obj_y0'], x1=r['obj_x1'], y1=r['obj_y1'], stage=r['stage'], source='legacy')
            for r in legacy]
    write(ANN, ANN_FIELDS, rows); write(IGN, IGN_FIELDS, [])
old = {r['photo']: r['label_status'] for r in read(MANIFEST)}
man = []
for f in photos:
    im = Image.open(PHOTOS / f); e = im.getexif(); w, h = im.size
    if e.get(274, 1) in (5, 6, 7, 8): w, h = h, w
    block, split = split_of(f[9:15])
    man.append(dict(photo=f, device='S25U' if 'Ultra' in str(e.get(272)) else 'S24FE', width=w, height=h,
                    block=block, split=split, label_status=old.get(f, 'partial')))
write(MANIFEST, MAN_FIELDS, man)

by = {r['photo']: r for r in man}; ann = read(ANN)
print('photos', len(man), 'annotations', len(ann))
for s in ('train', 'val', 'test'):
    ph = [m for m in man if m['split'] == s]
    c = C.Counter(a['stage'] for a in ann if by[a['photo']]['split'] == s)
    print(s, 'photos', len(ph), 'S25U', sum(m['device'] == 'S25U' for m in ph), dict(c), 'blocks', sorted({m['block'] for m in ph}))
