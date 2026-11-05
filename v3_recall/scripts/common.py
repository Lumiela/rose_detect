# v3 공용 경로와 입출력. 라벨은 이 폴더의 labels를 쓴다. v1_box/labels는 고정한 원본이고 val, test는 v1과 같아야 한다. 좌표는 EXIF 회전을 적용한 원본 사진 기준이다.
import csv, json, sys
from pathlib import Path

VER = Path(__file__).resolve().parents[1]
FLOWER = VER.parent
V1 = FLOWER / 'v1_box'
PHOTOS = FLOWER / 'common' / 'photos'
ANN, IGN, REJ, MANIFEST = (VER / 'labels' / n for n in ('annotations.csv', 'ignore_regions.csv', 'pseudo_rejected.csv', 'photos.csv'))
ANN_FIELDS = ['photo', 'ann_id', 'x0', 'y0', 'x1', 'y1', 'stage', 'source']; IGN_FIELDS = ['photo', 'x0', 'y0', 'x1', 'y1']
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


def assert_eval_same():
    # val, test 라벨과 제외 영역이 v1과 같은지 확인한다. 다르면 v1, v2 점수와 비교할 수 없으므로 멈춘다
    man = {m['photo']: m['split'] for m in read(MANIFEST)}
    for n, keys in (('annotations.csv', ANN_FIELDS), ('ignore_regions.csv', IGN_FIELDS)):
        rows = lambda p: sorted(tuple(r[k] for k in keys) for r in read(p / 'labels' / n) if man.get(r['photo']) in ('val', 'test'))
        assert rows(VER) == rows(V1), f'{n}: val, test 라벨이 v1과 다릅니다'
