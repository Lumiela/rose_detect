# 팀원이 브라우저로 여는 판정 파일을 사람별로 만든다. 설치와 서버 접속이 필요 없고, 칸 이미지까지 HTML 안에 들어 있다. 파란 박스가 판정 대상이다.
# 대상은 train 사진에서 third_pseudo가 찾았지만 라벨, 제외 영역과 겹치지 않는 예측이다. 사진 단위로 나눠 같은 사진은 한 사람이 본다. test와 val 사진은 넣지 않는다.
# 일부 사진은 두 명에게 주고(겹침), 사용자가 이미 판정한 train 칸도 표시 없이 섞는다(기준 칸). 번호와 칸의 관계는 labels/team_check_<회차>.csv에만 있다.
# 사용: python 59_make_team_files.py c1 [--conf 0.5] [--reviewers A,B,C,D,E] [--overlap 0.1] [--gold 0.03] [--limit N]
import argparse, base64, collections as C, io, json, random
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('round'); ap.add_argument('--conf', type=float, default=0.5); ap.add_argument('--reviewers', default='A,B,C,D,E'); ap.add_argument('--overlap', type=float, default=0.1); ap.add_argument('--gold', type=float, default=0.03); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--cell', type=int, default=200); ap.add_argument('--cols', type=int, default=10); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
L = VER / 'labels'; OUT = VER / 'runs/team_check' / a.round; OUT.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed); REV = a.reviewers.split(','); man = {m['photo']: m for m in read(MANIFEST)}


def ov(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h; x, y = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (x + y - i + 1e-9), i / (min(x, y) + 1e-9)


G, I = C.defaultdict(list), C.defaultdict(list)
for r in read(L / 'provenance.csv'):
    G[r['photo']].append([float(r[k]) for k in ('x0', 'y0', 'x1', 'y1')])
for r in read(IGN) + read(REJ):
    I[r['photo']].append(box(r))
P = json.loads((VER / 'runs/det/third_pseudo/cand/preds_train.json').read_text(encoding='utf-8')); cand = C.defaultdict(list)
for f, ps in P.items():
    assert man[f]['split'] == 'train'
    for p in ps:
        if p[4] >= a.conf and max([max(ov(p[:4], g)) for g in G[f]] or [0]) < 0.3 and not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in I[f]):
            cand[f].append(dict(photo=f, x0=round(p[0], 1), y0=round(p[1], 1), x1=round(p[2], 1), y1=round(p[3], 1), conf=round(p[4], 3)))
total = sum(len(v) for v in cand.values()); load = {r: 0 for r in REV}; own = C.defaultdict(list)
for f in sorted(cand, key=lambda f: (-len(cand[f]), f)):
    r = min(REV, key=lambda x: (load[x], x)); own[f].append(r); load[r] += len(cand[f])
ph = sorted(cand); rng.shuffle(ph); extra = 0
for f in ph:  # 겹침: 다른 한 사람에게도 준다
    if extra >= a.overlap * total:
        break
    if len(cand[f]) > 0.03 * total:
        continue
    r = min((x for x in REV if x not in own[f]), key=lambda x: (load[x], x)); own[f].append(r); load[r] += len(cand[f]); extra += len(cand[f])
# 기준 칸: 사용자가 판정한 train 칸. 봉오리와 잎을 반씩
E = {r['id']: r for r in read(L / 'review_train_errors_r3.csv')}; T = json.loads((L / 'review_train_fp.json').read_text(encoding='utf-8')); gold = {'봉오리': [], '아님': []}
for k, v in T.items():
    if v in ('봉오리', '잎') and man[E[k]['photo']]['split'] == 'train':
        gold['봉오리' if v == '봉오리' else '아님'].append(dict(photo=E[k]['photo'], x0=float(E[k]['x0']), y0=float(E[k]['y0']), x1=float(E[k]['x1']), y1=float(E[k]['y1']), conf='', gold='봉오리' if v == '봉오리' else '아님', gold_id=k))
UC = json.loads((L / 'user_check_r1.json').read_text(encoding='utf-8')); UR = read(L / 'user_check_r1.csv')
for r in UR:
    if UC.get(r['id']) == '아님':
        gold['아님'].append(dict(photo=r['photo'], x0=float(r['x0']), y0=float(r['y0']), x1=float(r['x1']), y1=float(r['y1']), conf='', gold='아님', gold_id=r['id']))
    elif UC.get(r['id']) == '맞음' and r['stage'] == '봉오리' and len(gold['봉오리']) < 60:
        gold['봉오리'].append(dict(photo=r['photo'], x0=float(r['x0']), y0=float(r['y0']), x1=float(r['x1']), y1=float(r['y1']), conf='', gold='봉오리', gold_id=r['id']))
for v in gold.values():
    rng.shuffle(v)
crop_cache = {}


def cell_img(c, size=None):
    size = size or a.cell; f = c['photo']; im = crop_cache.get(f)
    if im is None:
        crop_cache.clear(); im = crop_cache[f] = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB')
    b = [c['x0'], c['y0'], c['x1'], c['y1']]; s = max(b[2] - b[0], b[3] - b[1]) * 1.8 + 24; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; x0, y0 = cx - s / 2, cy - s / 2; k = size / s
    t = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((size, size), Image.BICUBIC); ImageDraw.Draw(t).rectangle([(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k], outline=(0, 120, 255), width=2)
    return t


def b64(im, q=82):
    o = io.BytesIO(); im.save(o, 'JPEG', quality=q); return base64.b64encode(o.getvalue()).decode()


# 예시: 사용자가 판정한 칸에서 고른다
stage = {r['id']: r for r in UR}; pick = lambda cond, n: rng.sample([r for r in UR if cond(r)], n)
EX = [(r, '봉오리') for r in pick(lambda r: UC.get(r['id']) == '맞음' and r['stage'] == '봉오리', 2)] + [(r, '개화중') for r in pick(lambda r: UC.get(r['id']) == '맞음' and r['stage'] == '개화중', 2)] + [(r, '만개') for r in pick(lambda r: UC.get(r['id']) == '맞음' and r['stage'] == '만개', 2)] + [(r, '아님') for r in pick(lambda r: UC.get(r['id']) == '아님', 2)] + [(r, '일부') for r in pick(lambda r: UC.get(r['id']) == '일부', 2)] + [(r, '알수없음') for r in pick(lambda r: UC.get(r['id']) == '알수없음', 2)]
examples = [dict(cap=cap, img=b64(cell_img(dict(photo=r['photo'], x0=float(r['x0']), y0=float(r['y0']), x1=float(r['x1']), y1=float(r['y1'])), 160), 80)) for r, cap in EX]
TPL = (VER / 'scripts/team_check_template.html').read_text(encoding='utf-8'); rows = []; used = 0
for r in REV:
    items = [dict(c, kind='candidate', gold='', gold_id='') for f in sorted(own) if r in own[f] for c in cand[f]]; n_g = round(a.gold * len(items)); gs = []
    for i in range(n_g):
        src = gold['아님'] if i % 2 else gold['봉오리']
        while src:
            g = src.pop()
            if not any(c['photo'] == g['photo'] and max(ov([c['x0'], c['y0'], c['x1'], c['y1']], [g['x0'], g['y0'], g['x1'], g['y1']])) >= 0.3 for c in items):
                gs.append(dict(g, kind='gold')); break
    items += gs; rng.shuffle(items); items = items[:a.limit] if a.limit else items; ids = [f'{r}{i + 1:03d}' for i in range(len(items))]
    S = Image.new('RGB', (a.cols * a.cell, ((len(items) + a.cols - 1) // a.cols) * a.cell), (40, 40, 40)); order = sorted(range(len(items)), key=lambda i: items[i]['photo'])
    for i in order:
        S.paste(cell_img(items[i]), ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
    data = dict(reviewer=r, round=a.round, cols=a.cols, cell=a.cell, cells=ids, examples=examples); html = TPL.replace('__REVIEWER__', r).replace('__SHEET__', b64(S)).replace('__DATA__', json.dumps(data, ensure_ascii=False)); name = f'꽃판정_{a.round}_{r}.html'; (OUT / name).write_text(html, encoding='utf-8')
    for i, c in zip(ids, items):
        rows.append(dict(cell=i, reviewer=r, kind=c['kind'], photo=c['photo'], x0=c['x0'], y0=c['y0'], x1=c['x1'], y1=c['y1'], conf=c['conf'], gold=c['gold'], gold_id=c['gold_id'], shared=len(own.get(c['photo'], []))))
    print(r, len(items), 'gold', len(gs), 'photos', len({c['photo'] for c in items}), name, round((OUT / name).stat().st_size / 1e6, 1), 'MB', flush=True)
if not a.limit:
    write(L / f'team_check_{a.round}.csv', list(rows[0]), rows)
print('MAKE_DONE total candidates', total, 'overlap cells', extra, flush=True)
