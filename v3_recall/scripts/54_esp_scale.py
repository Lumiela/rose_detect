# 학습 때의 봉오리 축척과 ESP 축척을 비교하고, 추론 축척을 맞췄을 때 봉오리 A, B, C를 찾는지 잰다. 재학습과 파일 변경은 없다.
# 사용: python 54_esp_scale.py
import collections as C, glob, time
from common import *
import paths  # noqa: F401
import numpy as np
from PIL import Image
from ultralytics import YOLO
from infer import detect, open_photo

OUT = VER / 'runs/esp_compare/low'; loc = {k: v['loc'] for k, v in json.loads((OUT / 'buds.json').read_text(encoding='utf-8')).items()}; man = {m['photo']: m for m in read(MANIFEST)}
FILES = sorted(glob.glob(str(FLOWER.parent / 'model_git/v10_farm_infer/esp_cam_20261103/inputs/*.jfif'))); m = YOLO(str(VER / 'runs/seg/A_base/weights/best.pt')); THR = 0.6
L = lambda r: max(box(r)[2] - box(r)[0], box(r)[3] - box(r)[1])
for s in ('train', 'val', 'test'):
    v = np.array([L(r) for r in read(ANN) if man[r['photo']]['split'] == s and r['stage'] == '봉오리']); print(s, '봉오리 라벨', len(v), '긴 변 25/50/75%', np.percentile(v, [25, 50, 75]).round(0).tolist(), '입력(타일 1280) 50%', round(float(np.median(v)) / 2, 1), flush=True)
pm = [json.loads(t) for t in (VER / 'labels/masks.jsonl').read_text(encoding='utf-8').splitlines() if '"pseudo"' in t]; v = np.array([max(x['box'][2] - x['box'][0], x['box'][3] - x['box'][1]) for x in pm]); print('임시 라벨', len(v), np.percentile(v, [25, 50, 75]).round(0).tolist(), flush=True)
print('사진 크기', C.Counter(f"{m_['device']} {m_['width']}x{m_['height']}" for m_ in man.values() if m_['split'] == 'train').most_common(4), flush=True)
rows = []
for name, tile, zoom in [('tile1280', 1280, 1), ('tile1100', 1100, 1), ('tile960', 960, 1), ('tile880', 880, 1), ('tile819', 819, 1), ('tile720', 720, 1), ('tile640', 640, 1), ('확대1.56+tile1280', 1280, 4000 / 2560), ('확대1.45+tile1280', 1280, 1.45)]:
    d = dict(A=0, B=0, C=0, nA=0, nB=0, nC=0, other=0, A50=0, B50=0, C50=0, other50=0); conf = C.defaultdict(list); t0 = time.time()
    for f in FILES:
        nm = Path(f).name[:6]; im = open_photo(f)
        if zoom != 1:
            im = im.resize((round(im.width * zoom), round(im.height * zoom)), Image.BICUBIC)
        P = detect(m, im, tile=tile)[0]; P = P.copy(); P[:, :4] /= zoom; near = set()
        for b, (x, y, s) in loc[nm].items():
            if s < 0.7:
                continue
            idx = [i for i, p in enumerate(P) if p[0] - 20 <= x <= p[2] + 20 and p[1] - 20 <= y <= p[3] + 20 and max(p[2] - p[0], p[3] - p[1]) < 220]; c = max((float(P[i][4]) for i in idx), default=0.0); d['n' + b] += 1; d[b] += c >= THR; d[b + '50'] += c >= 0.5; conf[b].append(round(c, 2)); near |= set(idx)
        d['other'] += sum(i not in near and P[i][4] >= THR for i in range(len(P))); d['other50'] += sum(i not in near and P[i][4] >= 0.5 for i in range(len(P)))
    sec = (time.time() - t0) / 6; rows.append(dict(name=name, tile=tile, zoom=zoom, A=d['A'], B=d['B'], C=d['C'], nA=d['nA'], nB=d['nB'], nC=d['nC'], other=d['other'], conf=dict(conf))); print(name, f"0.6: A {d['A']}/{d['nA']} B {d['B']}/{d['nB']} C {d['C']}/{d['nC']} 합계 {d['A'] + d['B'] + d['C']}/16 다른 {d['other']} | 0.5: {d['A50'] + d['B50'] + d['C50']}/16 다른 {d['other50']} | {sec:.2f}s", dict(conf), flush=True)
(OUT / 'scale.json').write_text(json.dumps(rows, ensure_ascii=False, default=int), encoding='utf-8'); print('SCALE_DONE', flush=True)
