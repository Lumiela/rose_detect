# 평가 사진 3장에서 v3와 파인튜닝 모델의 결과를 그림으로 비교하고, 임계값에 따른 변화를 참고용으로 출력한다. 임계값은 val에서 고정한 값이고 이 출력으로 바꾸지 않는다. 사용: python 33_esp_figs.py
from common import *
from PIL import Image, ImageDraw, ImageFont
OUT = VER / 'runs' / 'report'; OUT.mkdir(parents=True, exist_ok=True); pred = json.loads((VER / 'runs/eval/preds_test.json').read_text()); sel = json.loads((VER / 'runs/eval/selected.json').read_text())
LAB = {}
for t in (VER / 'labels' / 'masks_body.jsonl').read_text(encoding='utf-8').splitlines():
    r = json.loads(t); LAB.setdefault(r['photo'], []).append(r)
PH = sorted(pred['v3_t880']); font = ImageFont.load_default(size=26)


def match(q, L):
    cx, cy = (q[0] + q[2]) / 2, (q[1] + q[3]) / 2
    return next((i for i, r in enumerate(L) if r['box'][0] <= cx <= r['box'][2] and r['box'][1] <= cy <= r['box'][3]), None)


def count(name, t):
    f = o = 0
    for p in PH:
        u = set()
        for q in pred[name][p]:
            if q[4] < t:
                continue
            i = match(q, LAB[p])
            if i is None:
                o += 1
            elif i not in u:
                u.add(i); f += 1
    return f, o


for n in pred:
    print(n, {t: count(n, t) for t in (.2, .3, .4, .5, .6, .7, .75)}, flush=True)
Rows = [('v3_t1280', .6, 'v3 tile 1280, conf .60'), ('v3_t880', .6, 'v3 tile 880, conf .60'), ('ft_s1_t880', sel['ft_s1_t880'], f"fine-tuned seed 1, tile 880, conf {sel['ft_s1_t880']:.2f}")]
W, H = 800, 600; S = Image.new('RGB', (W * 3, (H + 40) * len(Rows)), (20, 20, 20)); D = ImageDraw.Draw(S); k = W / 2560
for ri, (n, t, title) in enumerate(Rows):
    D.text((8, ri * (H + 40) + 4), title, fill=(255, 230, 120), font=font)
    for ci, p in enumerate(PH):
        im = Image.open(PHOTOS / p).convert('RGB').resize((W, H)); d = ImageDraw.Draw(im); L = LAB[p]; u = set()
        for q in pred[n][p]:
            if q[4] < t:
                continue
            i = match(q, L)
            if i is not None:
                u.add(i)
            c = (60, 255, 90) if i is not None else (255, 190, 0); d.rectangle([q[0] * k - 6, q[1] * k - 6, q[2] * k + 6, q[3] * k + 6], outline=c, width=4); d.text((q[0] * k, q[1] * k - 34), f'{L[i]["bud"] if i is not None else "?"} {q[4]:.2f}', fill=c, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        for i, r in enumerate(L):
            if i not in u:
                b = r['tight']; d.ellipse([(b[0] + b[2]) / 2 * k - 26, (b[1] + b[3]) / 2 * k - 26, (b[0] + b[2]) / 2 * k + 26, (b[1] + b[3]) / 2 * k + 26], outline=(255, 50, 50), width=4); d.text(((b[0] + b[2]) / 2 * k + 28, (b[1] + b[3]) / 2 * k - 16), r['bud'], fill=(255, 50, 50), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        S.paste(im, (ci * W, ri * (H + 40) + 40))
S.save(OUT / 'test_compare.jpg', quality=88); print('FIG_DONE', flush=True)
