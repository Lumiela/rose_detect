# v2 공용 경로와 입출력. 박스 라벨 원본은 v1_box/labels 한 곳을 함께 쓴다. 좌표는 EXIF 회전을 적용한 원본 사진 기준이다.
import csv, json, sys
from pathlib import Path

VER = Path(__file__).resolve().parents[1]
FLOWER = VER.parent
V1 = FLOWER / 'v1_box'
PHOTOS = FLOWER / 'common' / 'photos'
ANN, IGN, REJ, MANIFEST = (V1 / 'labels' / n for n in ('annotations.csv', 'ignore_regions.csv', 'pseudo_rejected.csv', 'photos.csv'))
PSEUDO_PREDS = V1 / 'runs' / 'det' / 'second' / 'cand' / 'preds_train.json'  # v1 third_pseudo와 같은 임시 라벨의 출처
MASKS = VER / 'labels' / 'masks.jsonl'
STAGES = ['봉오리', '개화중', '만개']
sys.path.insert(0, str(FLOWER / 'common'))


def read(path):
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def box(r):
    return [float(r[k]) for k in ('x0', 'y0', 'x1', 'y1')]


def ov(p, q):
    # (IoU, 작은 박스 대비 겹침)
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    x, y = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (x + y - i + 1e-9), i / (min(x, y) + 1e-9)


def read_masks(path=MASKS):
    return [json.loads(t) for t in Path(path).read_text(encoding='utf-8').splitlines() if t.strip()]
