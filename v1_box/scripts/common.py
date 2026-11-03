# 공용 경로와 라벨 입출력. 원본 사진 좌표는 EXIF 회전을 적용한 기준이다.
import csv, sys
from pathlib import Path

VER = Path(__file__).resolve().parents[1]
FLOWER = VER.parent  # 꽃 모델 폴더. 버전이 함께 쓰는 사진은 common/에 둔다
ROOT = FLOWER.parent  # 저장소 루트. 잎 모델과 함께 쓰는 common/, weights/가 있다
FARM = ROOT / '장미농장'
PHOTOS = FARM / '미니장미 사진'
LEGACY_CSV = PHOTOS / '꽃봉오리' / 'labels.csv'
if (FLOWER / 'common' / 'photos').is_dir():  # 서버
    PHOTOS = FLOWER / 'common' / 'photos'
    LEGACY_CSV = FLOWER / 'common' / 'labels_20261030.csv'
ANN = VER / 'labels' / 'annotations.csv'
IGN = VER / 'labels' / 'ignore_regions.csv'
REJ = VER / 'labels' / 'pseudo_rejected.csv'  # 사람이 꽃이 아니라고 판정한 임시 라벨
MANIFEST = VER / 'labels' / 'photos.csv'
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
