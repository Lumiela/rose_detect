# 상자를 그대로 넣으면 SAM2.1이 꽃받침과 잎까지 잡아서, 프롬프트를 여러 개 넣고 가는 부분을 지운 뒤 가장 둥근 후보를 봉오리 몸통으로 고른다. 라벨 원본은 고치지 않는다.
# 사용: python 11_esp_body.py
from common import *
import paths  # noqa: F401
import cv2, numpy as np
from PIL import Image, ImageDraw
from ultralytics import SAM
model = SAM(str(paths.WEIGHTS / 'sam2.1_s.pt')); R = raw(); SP = splits(); out = []


def post(m, x0, y0, b):
    w, h = b[2] - b[0], b[3] - b[1]; k = max(3, int(.2 * min(w, h))) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    c = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); m = (lab == c).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cn = max(cs, key=cv2.contourArea); A = cv2.contourArea(cn); P = cv2.arcLength(cn, True)
    if A < 20 or P == 0:
        return None
    hull = cv2.contourArea(cv2.convexHull(cn)); circ = min(1, 4 * np.pi * A / (P * P)); sol = A / max(hull, 1); ys, xs = np.nonzero(m); t = [int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0 + 1), int(ys.max() + y0 + 1)]; ratio = float(m.sum()) / (w * h)
    poly = cv2.approxPolyDP(cn, 1.0, True).reshape(-1, 2) + [x0, y0]
    return dict(area=int(m.sum()), ratio=round(ratio, 3), circ=round(float(circ), 3), solidity=round(float(sol), 3), tight=t, poly=poly.tolist()) if len(poly) >= 3 else None


def cands(img, b):
    H, W = img.shape[:2]; w, h = b[2] - b[0], b[3] - b[1]; pad = max(w, h) * .6 + 12; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0, y0, x1, y1 = max(0, int(b[0] - pad)), max(0, int(b[1] - pad)), min(W, int(b[2] + pad + 1)), min(H, int(b[3] + pad + 1)); crop = np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]); S = lambda f: [cx - w * f / 2 - x0, cy - h * f / 2 - y0, cx + w * f / 2 - x0, cy + h * f / 2 - y0]
    prompts = [('box', dict(bboxes=[S(1)])), ('box.8', dict(bboxes=[S(.8)])), ('box.6', dict(bboxes=[S(.6)])), ('pt', dict(points=[[cx - x0, cy - y0]], labels=[1])), ('box.8+pt', dict(bboxes=[S(.8)], points=[[cx - x0, cy - y0]], labels=[1]))]
    for dx, dy in ((-.2, 0), (.2, 0), (0, -.2), (0, .2)):
        prompts.append((f'pt{dx}{dy}', dict(points=[[cx + w * dx - x0, cy + h * dy - y0]], labels=[1])))
    res = []
    for name, kw in prompts:
        r = model(crop, verbose=False, **kw)[0]
        if r.masks is None or not len(r.masks.data):
            continue
        for mm in r.masks.data:
            m = mm.cpu().numpy().astype(np.uint8)
            if m.shape != crop.shape[:2]:
                m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            q = post(m, x0, y0, b)
            if q is None or not .12 <= q['ratio'] <= .8:
                continue
            t = q['tight']
            if t[0] < b[0] - 12 or t[1] < b[1] - 12 or t[2] > b[2] + 12 or t[3] > b[3] + 12:
                continue
            q['prompt'] = name; q['score'] = round(q['circ'] * q['solidity'], 3); res.append(q)
    return res


for p in sorted(R):
    img = np.asarray(Image.open(PHOTOS / p).convert('RGB'))
    for b in R[p]['boxes']:
        bx = [b['x0'], b['y0'], b['x1'], b['y1']]; C = cands(img, bx); best = max(C, key=lambda q: q['score']) if C else None
        out.append(dict(photo=p, split=SP[p], bud=BUD[p[:6]][b['id']], label_id=b['id'], stage=b['stage'], box=bx, n=len(C), **(best or dict(poly=None))))
(VER / 'labels' / 'masks_body.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')
W = 230; order = sorted(out, key=lambda r: (r['bud'], r['photo'])); cols = 7; rows = (len(order) + cols - 1) // cols; S = Image.new('RGB', (W * cols, (W + 18) * rows), (25, 25, 25)); D = ImageDraw.Draw(S)
for i, r in enumerate(order):
    im = Image.open(PHOTOS / r['photo']).convert('RGB'); b = r['box']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; h = max(b[2] - b[0], b[3] - b[1]) * .75 + 12; k = W / (2 * h)
    c = im.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((W, W), Image.LANCZOS); d = ImageDraw.Draw(c); T = lambda x, y: ((x - cx + h) * k, (y - cy + h) * k)
    if r['poly']:
        d.line([T(*q) for q in r['poly']] + [T(*r['poly'][0])], fill=(60, 255, 90), width=2); t = r['tight']; d.rectangle([T(t[0], t[1]), T(t[2], t[3])], outline=(60, 170, 255), width=1)
    x, y = (i % cols) * W, (i // cols) * (W + 18); S.paste(c, (x, y + 18)); tw = (r['tight'][2] - r['tight'][0], r['tight'][3] - r['tight'][1]) if r['poly'] else 'NONE'
    D.text((x + 3, y + 3), f"{i} {r['bud']} {r['photo'][:6]} {r['split'][:2]} {tw} {r.get('score','')}", fill=(255, 230, 120))
S.save(VER / 'runs' / 'qa' / 'masks_body.jpg', quality=88); print('BODY_DONE', len(out), 'none', sum(r['poly'] is None for r in out), flush=True)
