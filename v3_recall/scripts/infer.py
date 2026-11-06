# 사진 한 장에서 꽃을 찾고 단계를 분류한다. 31_eval_seg.py와 40_flower_ids.py가 함께 쓴다. 좌표는 EXIF 회전을 적용한 원본 사진 기준이다.
from common import *
import paths  # noqa: F401  (opencv 우회가 먼저)
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU가 없으면 CPU로 돈다. 결과는 같고 느리다
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def open_photo(path):
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def starts(size, tile, overlap):
    if size <= tile:
        return [0]
    return list(range(0, size - tile, int(tile * (1 - overlap)))) + [size - tile]


def iou_ios(b, B):
    x0, y0 = np.maximum(b[0], B[:, 0]), np.maximum(b[1], B[:, 1]); x1, y1 = np.minimum(b[2], B[:, 2]), np.minimum(b[3], B[:, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None); ab = (b[2] - b[0]) * (b[3] - b[1]); aB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return i / (ab + aB - i + 1e-9), i / (np.minimum(ab, aB) + 1e-9)


def merge(P):
    # P 열: x0 y0 x1 y1 conf 조각 번호. 조각은 사진 가장자리가 아닌 타일 경계에 닿은 박스다
    whole = P[P[:, 5] == 0, :4]
    if len(whole):
        P = np.array([p for p in P if not p[5] or iou_ios(p[:4], whole)[1].max() < 0.5]).reshape(-1, 7)
    keep = []
    for p in P[np.argsort(-P[:, 4])]:
        if keep:
            io, os_ = iou_ios(p[:4], np.array(keep)[:, :4])
            if (io > 0.5).any() or (os_ > 0.8).any():
                continue
        keep.append(p)
    return np.array(keep).reshape(-1, 7)


def detect(model, im, tile=1280, overlap=0.25, imgsz=640, batch=32, mask_box=False):
    # (박스와 conf의 N×5 배열, 다각형 N개). conf 높은 순서이고 임계값은 호출하는 쪽에서 적용한다
    views = []; side = max(im.size); P = []; polys = []
    for y in starts(im.height, tile, overlap):
        for x in starts(im.width, tile, overlap):
            c = Image.new('RGB', (tile, tile), (114, 114, 114)); c.paste(im.crop((x, y, min(x + tile, im.width), min(y + tile, im.height))), (0, 0))
            views.append((c.resize((imgsz, imgsz), Image.BILINEAR), x, y, tile / imgsz))
    c = Image.new('RGB', (side, side), (114, 114, 114)); c.paste(im, (0, 0)); views.append((c.resize((imgsz, imgsz), Image.BILINEAR), 0, 0, side / imgsz))
    for i in range(0, len(views), batch):
        vs = views[i:i + batch]
        for v, r in zip(vs, model.predict([t[0] for t in vs], imgsz=imgsz, conf=0.05, iou=0.6, retina_masks=True, verbose=False)):
            if r.masks is None or not len(r.boxes):
                continue
            t = r.boxes.xyxy.cpu().numpy(); b = t * v[3]; b[:, [0, 2]] += v[1]; b[:, [1, 3]] += v[2]; xy = [q * v[3] + [v[1], v[2]] for q in r.masks.xy]
            if mask_box:
                for j, q in enumerate(xy):
                    if len(q):
                        b[j] = [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
            full = v[3] != tile / imgsz; m = 2
            frag = np.zeros(len(b), bool) if full else ((t[:, 0] <= m) & (v[1] > 0)) | ((t[:, 1] <= m) & (v[2] > 0)) | ((t[:, 2] >= imgsz - m) & (v[1] + tile < im.width)) | ((t[:, 3] >= imgsz - m) & (v[2] + tile < im.height))
            P.append(np.c_[b, r.boxes.conf.cpu().numpy(), frag, np.arange(len(polys), len(polys) + len(b))]); polys += xy
    if not P:
        return np.zeros((0, 5)), []
    M = merge(np.vstack(P)); M[:, [0, 2]] = M[:, [0, 2]].clip(0, im.width); M[:, [1, 3]] = M[:, [1, 3]].clip(0, im.height)
    return M[:, :5], [polys[int(k)] for k in M[:, 6]]


def load_cls(path):
    net = efficientnet_v2_s(); net.classifier[1] = nn.Linear(net.classifier[1].in_features, 3); net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.to(DEV).eval()


def crop_box(p):
    # 분류기 입력 영역: 박스 긴 변 × 1.25 정사각형
    s = max(p[2] - p[0], p[3] - p[1]) * 1.25; cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)


def classify(net, im, P):
    # 박스마다 STAGES 순서의 확률. N×3 배열
    if not len(P):
        return np.zeros((0, 3))
    xs = [TF(im.crop(crop_box(p))) for p in P]; out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
        for i in range(0, len(xs), 128):
            out.append(net(torch.stack(xs[i:i + 128]).to(DEV)).float().softmax(1).cpu().numpy())
    return np.vstack(out)
