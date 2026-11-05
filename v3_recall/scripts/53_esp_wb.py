# 새 ESP 6장의 봉오리 A, B, C를 색 보정(잎 모델의 gray_world)과 입력 960 학습 모델(B_960)로 다시 잰다. 재학습과 파일 변경은 없다. 기존 큐브 사진의 단계 분포도 보정 전후로 비교한다.
# 사용: python 53_esp_wb.py
import collections as C, glob, sys
from common import *
import paths  # noqa: F401
import numpy as np
from PIL import Image
from ultralytics import YOLO
from infer import detect, open_photo, load_cls, classify
sys.path.insert(0, str(FLOWER.parent / 'model_git/v9_fastinfer/deploy'))
from leafdx.stage2 import gray_world

OUT = VER / 'runs/esp_compare/low'; loc = {k: v['loc'] for k, v in J(OUT / 'buds.json').items()} if False else {k: v['loc'] for k, v in json.loads((OUT / 'buds.json').read_text(encoding='utf-8')).items()}
FILES = sorted(glob.glob(str(FLOWER.parent / 'model_git/v10_farm_infer/esp_cam_20261103/inputs/*.jfif'))); CUBE = sorted(p for p in glob.glob(str(V1 / 'cube/**/*.jpg'), recursive=True)); THR = dict(v3=0.60, B=0.65); M = dict(v3=YOLO(str(VER / 'runs/seg/A_base/weights/best.pt')), B=YOLO(str(VER / 'runs/seg/B_960/weights/best.pt'))); net = load_cls(V1 / 'runs/cls/effnetv2s_r2/best.pt'); ST = ['봉오리', '개화중', '만개']; SZ = dict(v3=640, B=960); tot = {}


def wb(im):
    return Image.fromarray(gray_world(np.asarray(im)).astype(np.uint8) if True else None)


for f in FILES:
    nm = Path(f).name[:6]; im0 = open_photo(f); im1 = wb(im0); a0, a1 = np.asarray(im0).reshape(-1, 3).mean(0), np.asarray(im1).reshape(-1, 3).mean(0); print(nm, '평균 RGB', a0.round(0).tolist(), '->', a1.round(0).tolist(), flush=True)
    if nm == '0d68e2':
        S = Image.new('RGB', (1800, 675)); S.paste(im0.resize((900, 675)), (0, 0)); S.paste(im1.resize((900, 675)), (900, 0)); S.save(OUT / 'wb_compare.jpg', quality=85)
    for mk, m in M.items():
        for tl in (1280, 960):
            for cn, im in (('raw', im0), ('wb', im1)):
                P = detect(m, im, tile=tl, imgsz=SZ[mk])[0]; thr = THR[mk]; key = (mk, tl, cn); d = tot.setdefault(key, dict(A=0, B=0, C=0, nA=0, nB=0, nC=0, other=0, conf=C.defaultdict(list), stage=C.Counter())); near = set(); Q = P[P[:, 4] >= thr]; pr = classify(net, im, Q[:, :4]) if len(Q) else np.zeros((0, 3))
                for b, (x, y, s) in loc[nm].items():
                    if s < 0.7:
                        continue
                    idx = [i for i, p in enumerate(P) if p[0] - 20 <= x <= p[2] + 20 and p[1] - 20 <= y <= p[3] + 20 and max(p[2] - p[0], p[3] - p[1]) < 220]; c = max((float(P[i][4]) for i in idx), default=0.0); d['n' + b] += 1; d[b] += c >= thr; d['conf'][b].append(round(c, 2))
                    for i in idx:
                        near.add(tuple(P[i][:4]))
                    for j, q in enumerate(Q):
                        if tuple(q[:4]) in near and len(pr):
                            d['stage'][ST[int(pr[j].argmax())]] += 0
                d['other'] += sum(tuple(q[:4]) not in near for q in Q)
                for j, q in enumerate(Q):
                    if tuple(q[:4]) in near:
                        d['stage'][ST[int(pr[j].argmax())]] += 1
print('\n## 봉오리별 (찾은 횟수/볼 수 있는 사진), 다른 예측')
for k, d in tot.items():
    print(k, f"A {d['A']}/{d['nA']} B {d['B']}/{d['nB']} C {d['C']}/{d['nC']} 합계 {d['A'] + d['B'] + d['C']}/{d['nA'] + d['nB'] + d['nC']} 다른 예측 {d['other']}", dict(d['conf']), dict(d['stage']), flush=True)
print('\n## 기존 큐브 69장 (v3, 타일 1280)')
for cn in ('raw', 'wb', 'wb_cls_only'):
    st = C.Counter(); n = 0
    for f in CUBE:
        im0 = open_photo(f); im1 = wb(im0); P = detect(M['v3'], im1 if cn == 'wb' else im0)[0]; Q = P[P[:, 4] >= 0.6]; n += len(Q)
        if len(Q):
            st.update(ST[int(i)] for i in classify(net, im0 if cn == 'raw' else im1, Q[:, :4]).argmax(1))
    print(cn, n, {k: f'{st[k]} ({st[k] / max(n, 1):.1%})' for k in ST}, flush=True)
print('WB_DONE', flush=True)
