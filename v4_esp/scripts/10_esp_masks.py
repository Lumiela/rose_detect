# 사람이 그린 상자를 프롬프트로 SAM2.1 마스크를 만들고, 마스크에 맞는 좁은 상자를 얻는다. 라벨 원본은 고치지 않는다. 사용: python 10_esp_masks.py
from common import *
import paths  # noqa: F401
import cv2, numpy as np
from PIL import Image, ImageDraw
from ultralytics import SAM
model = SAM(str(paths.WEIGHTS / 'sam2.1_s.pt')); R = raw(); SP = splits(); out = []


def segment(img, b, pad_ratio=0.75):
    H, W = img.shape[:2]; pad = max(b[2] - b[0], b[3] - b[1]) * pad_ratio + 16
    x0, y0, x1, y1 = max(0, int(b[0] - pad)), max(0, int(b[1] - pad)), min(W, int(b[2] + pad + 1)), min(H, int(b[3] + pad + 1))
    crop = np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]); r = model(crop, bboxes=[[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]], verbose=False)[0]
    if r.masks is None or not len(r.masks.data):
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8); sc = float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes) else 0.0
    if m.shape != crop.shape[:2]:
        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); m = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cs, key=cv2.contourArea); poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + [x0, y0]
    ys, xs = np.nonzero(m)
    return dict(score=round(sc, 3), area=int(m.sum()), tight=[int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0 + 1), int(ys.max() + y0 + 1)], parts=int(n - 1), poly=poly.tolist())


for p in sorted(R):
    img = np.asarray(Image.open(PHOTOS / p).convert('RGB'))
    for b in R[p]['boxes']:
        bx = [b['x0'], b['y0'], b['x1'], b['y1']]; q = segment(img, bx)
        out.append(dict(photo=p, split=SP[p], bud=BUD[p[:6]][b['id']], label_id=b['id'], stage=b['stage'], box=bx, **(q or dict(poly=None))))
MASKS.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')
# 검수 시트
W = 230; order = sorted(out, key=lambda r: (r['bud'], r['photo'])); cols = 7; rows = (len(order) + cols - 1) // cols; S = Image.new('RGB', (W * cols, (W + 18) * rows), (25, 25, 25)); D = ImageDraw.Draw(S)
for i, r in enumerate(order):
    im = Image.open(PHOTOS / r['photo']).convert('RGB'); b = r['box']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; h = max(b[2] - b[0], b[3] - b[1]) * .75 + 12; k = W / (2 * h)
    c = im.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((W, W), Image.LANCZOS); d = ImageDraw.Draw(c); T = lambda x, y: ((x - cx + h) * k, (y - cy + h) * k)
    if r['poly']:
        d.line([T(*q) for q in r['poly']] + [T(*r['poly'][0])], fill=(60, 255, 90), width=2); t = r['tight']; d.rectangle([T(t[0], t[1]), T(t[2], t[3])], outline=(60, 170, 255), width=2)
    x, y = (i % cols) * W, (i // cols) * (W + 18); S.paste(c, (x, y + 18)); tw = (r['tight'][2] - r['tight'][0], r['tight'][3] - r['tight'][1]) if r['poly'] else ''
    D.text((x + 3, y + 3), f"{r['bud']} {r['photo'][:6]} {r['split'][:2]} {tw}", fill=(255, 230, 120))
(VER / 'runs' / 'qa').mkdir(parents=True, exist_ok=True); S.save(VER / 'runs' / 'qa' / 'masks_e1.jpg', quality=88)
print('MASK_DONE', len(out), 'empty', sum(r['poly'] is None for r in out), flush=True)
for r in order:
    print(r['bud'], r['photo'][:6], r['split'], 'box', round(r['box'][2] - r['box'][0]), round(r['box'][3] - r['box'][1]), 'tight', (r['tight'][2] - r['tight'][0], r['tight'][3] - r['tight'][1]) if r['poly'] else None, 'score', r.get('score'), 'parts', r.get('parts'))
