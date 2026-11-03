# 세그멘테이션 라벨 확인: 내보낸 타일에 다각형을 그린다.
# 사용: python 21_check_labels.py <출력 폴더> [--split train] [--n 12] [--seed 0]
import argparse, random
from common import *
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--data', default=str(VER / 'data')); ap.add_argument('--split', default='train'); ap.add_argument('--n', type=int, default=12); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
D = Path(a.data); fs = sorted((D / 'images' / a.split).glob('*.jpg')); random.Random(a.seed).shuffle(fs); fs = [f for f in fs if (D / 'labels' / a.split / (f.stem + '.txt')).read_text().strip()][:a.n]; ims = []
for f in fs:
    im = Image.open(f).convert('RGB'); d = ImageDraw.Draw(im); S = im.width
    for t in (D / 'labels' / a.split / (f.stem + '.txt')).read_text().splitlines():
        v = [float(x) * S for x in t.split()[1:]]; pts = list(zip(v[0::2], v[1::2])); d.line(pts + pts[:1], fill=(0, 255, 255), width=3)
    ims.append(im)
c = 3; n = (len(ims) + c - 1) // c; s = Image.new('RGB', (c * 640, n * 640))
for i, im in enumerate(ims):
    s.paste(im, ((i % c) * 640, (i // c) * 640))
Path(a.out).mkdir(parents=True, exist_ok=True); s.resize((s.width // 2, s.height // 2)).save(Path(a.out) / 'labels_check.jpg', quality=85); print('checked', len(ims))
