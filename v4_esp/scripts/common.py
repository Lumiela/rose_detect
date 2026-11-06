# v4 공용 경로. 라벨 원본은 labels/esp_e1_raw.json이고 고치지 않는다. 번호 대응은 Claude가 생김새로 맞춘 값이다.
import csv, json, sys
from pathlib import Path
VER = Path(__file__).resolve().parents[1]; FLOWER = VER.parent; V3 = FLOWER / 'v3_recall'
PHOTOS = VER / 'photos'; RAW = VER / 'labels' / 'esp_e1_raw.json'; SPLIT = VER / 'labels' / 'split.csv'; MASKS = VER / 'labels' / 'masks.jsonl'
sys.path.insert(0, str(FLOWER / 'common'))
BUD = {'040525': {1: 'B', 2: 'A', 3: 'C'}, '0d68e2': {1: 'A', 2: 'C', 3: 'B'}, '42e5f5': {1: 'C', 2: 'A', 3: 'B'}, '795edd': {1: 'A', 2: 'C', 3: 'B'}, '817a7c': {1: 'A', 2: 'C', 3: 'B'}, 'b676fe': {1: 'B', 2: 'A', 3: 'C'},
       '135225': {1: 'R', 2: 'G', 3: 'P'}, '1859c6': {1: 'P', 2: 'G', 3: 'R'}, '5923b8': {1: 'G', 2: 'R'}, '9700ce': {1: 'R', 2: 'G'}, '984aa8': {1: 'R', 2: 'G'}, 'ab06bd': {1: 'R', 2: 'G'},
       '260eac': {1: 'R', 2: 'G', 3: 'P'}, '804b80': {1: 'R', 2: 'G', 3: 'P'}, '6964dd': {1: 'P', 2: 'G', 3: 'R'}, '8583c9': {1: 'G', 2: 'R', 3: 'P'}, 'eee390': {1: 'R', 2: 'G', 3: 'P'}}


def splits():
    with open(SPLIT, encoding='utf-8-sig', newline='') as f:
        return {r['photo']: r['split'] for r in csv.DictReader(f)}


def raw():
    return json.loads(RAW.read_text(encoding='utf-8'))['photos']


def masks():
    return [json.loads(t) for t in MASKS.read_text(encoding='utf-8').splitlines() if t.strip()]
