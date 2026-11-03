# 서버로 올릴 묶음을 만든다. 사진, 크롭, 큐브 사진, 라벨, 스크립트를 담고 압축은 하지 않는다.
# 사용: python 30_package.py [출력 경로, 기본 upload.zip]
import sys, zipfile
from common import *

out = Path(sys.argv[1]) if len(sys.argv) > 1 else VER / 'upload.zip'
tmp = out.with_suffix('.tmp'); n = 0
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
    for p in sorted(PHOTOS.glob('*.jpg')):
        z.write(p, 'photos/' + p.name); n += 1
    fb = PHOTOS / '꽃봉오리'
    z.write(fb / 'labels.csv', 'labels_20261030.csv')
    for s in STAGES:
        for p in sorted((fb / s).glob('*.jpg')):
            z.write(p, f'crops/{s}/{p.name}'); n += 1
    for p in sorted((FARM / '스마트큐브사진_2026-10-30').glob('*/*.jpg')):
        z.write(p, f'cube/{p.parent.name}/{p.name}'); n += 1
    for base in ('labels', 'scripts'):
        for p in sorted((VER / base).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p, p.relative_to(VER).as_posix()); n += 1
    z.write(VER / 'README.md', 'README.md')
tmp.replace(out); print(out, n, 'files', out.stat().st_size)
