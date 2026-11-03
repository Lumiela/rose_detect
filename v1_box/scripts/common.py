# v11 공용 경로와 라벨 입출력. 원본 사진 좌표는 EXIF 회전을 적용한 기준이다.
import csv, sys
from pathlib import Path

V11 = Path(__file__).resolve().parents[1]
FARM = V11.parent / '장미농장'
PHOTOS = FARM / '미니장미 사진'
LEGACY_CSV = PHOTOS / '꽃봉오리' / 'labels.csv'
if (V11 / 'photos').is_dir():  # 서버: 30_package.py 묶음을 푼 구조
    PHOTOS = V11 / 'photos'
    LEGACY_CSV = V11 / 'labels_20261030.csv'
ANN = V11 / 'labels' / 'annotations.csv'
IGN = V11 / 'labels' / 'ignore_regions.csv'
MANIFEST = V11 / 'labels' / 'photos.csv'
STAGES = ['봉오리', '개화중', '만개']
ANN_FIELDS = ['photo', 'ann_id', 'x0', 'y0', 'x1', 'y1', 'stage', 'source']
IGN_FIELDS = ['photo', 'x0', 'y0', 'x1', 'y1']
MAN_FIELDS = ['photo', 'device', 'width', 'height', 'block', 'split', 'label_status']
# label_status: partial(기존 박스만) / draft(전체 라벨 초안) / verified(사람 검수 완료)
COMPLETE = {'draft', 'verified'}


def read(path):
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def box(r):
    return [float(r[k]) for k in ('x0', 'y0', 'x1', 'y1')]
