# 초기 판정(c1)에서 한 명만 본 칸을 다른 한 사람에게 더 주는 추가 판정 파일을 만든다. 모든 후보 칸을 두 명이 보게 하고, 다섯 명의 모든 짝이 고르게 겹치도록 사진 단위로 배정한다.
# 번호는 새로 매기고 1차 판정은 보여 주지 않는다. 기준 칸은 그 사람이 1차에서 본 적 없는 칸만 섞는다. 번호와 칸의 관계는 labels/team_check_<회차>.csv에만 있고, 라벨은 바꾸지 않는다.
# 사용: python 61_make_team_cross.py c2 --base c1 --out ../human_label/c2_cross/html [--gold 0.03] [--seed 0]
import argparse, base64, collections as C, io, json, random, re
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('round'); ap.add_argument('--base', default='c1'); ap.add_argument('--out', required=True); ap.add_argument('--gold', type=float, default=0.03); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--cell', type=int, default=200); ap.add_argument('--cols', type=int, default=10); a = ap.parse_args()
L = VER / 'labels'; OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed); man = {m['photo']: m for m in read(MANIFEST)}; base = read(L / f'team_check_{a.base}.csv'); REV = sorted({r['reviewer'] for r in base})
key = lambda r: (r['photo'], r['x0'], r['y0'], r['x1'], r['y1'])


def ov(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h; x, y = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (x + y - i + 1e-9), i / (min(x, y) + 1e-9)


xy = lambda r: [float(r[k]) for k in ('x0', 'y0', 'x1', 'y1')]
boxes = C.defaultdict(list); first = {}
for r in base:
    if r['kind'] == 'candidate':
        boxes[key(r)].append(r['reviewer']); first[key(r)] = r
photos = C.defaultdict(list)
for k, rs in boxes.items():
    photos[k[0]].append(k)
total = len(boxes); pair = C.Counter(); load = C.Counter(); own = {}
for f, ks in photos.items():
    who = {tuple(sorted(boxes[k])) for k in ks}; assert len(who) == 1, f; own[f] = list(next(iter(who))); assert man[f]['split'] == 'train', f
for k, rs in boxes.items():
    for r in rs:
        load[r] += 1
    if len(rs) == 2:
        pair[tuple(sorted(rs))] += 1
T_PAIR = total / (len(REV) * (len(REV) - 1) / 2); T_LOAD = total * 2 / len(REV); assign = {}; extra = C.defaultdict(list)
for f in sorted((f for f in photos if len(own[f]) == 1), key=lambda f: (-len(photos[f]), f)):
    r = own[f][0]; n = len(photos[f])
    s = min((x for x in REV if x != r), key=lambda x: ((pair[tuple(sorted((r, x)))] + n) / T_PAIR + (load[x] + n) / T_LOAD, x))
    assign[f] = s; pair[tuple(sorted((r, s)))] += n; load[s] += n; extra[s] += photos[f]
# 기준 칸: 1차에서 쓰지 않은 칸을 먼저 쓰고, 모자라면 다른 사람의 1차 기준 칸을 쓴다
E = {r['id']: r for r in read(L / 'review_train_errors_r3.csv')}; T = json.loads((L / 'review_train_fp.json').read_text(encoding='utf-8')); pool = {'봉오리': [], '아님': []}


def g_row(src, gid, v):
    return dict(photo=src['photo'], x0=src['x0'], y0=src['y0'], x1=src['x1'], y1=src['y1'], conf='', gold=v, gold_id=gid)


for k, v in T.items():
    if v in ('봉오리', '잎') and man[E[k]['photo']]['split'] == 'train':
        pool['봉오리' if v == '봉오리' else '아님'].append(g_row(E[k], k, '봉오리' if v == '봉오리' else '아님'))
UC = json.loads((L / 'user_check_r1.json').read_text(encoding='utf-8'))
for r in read(L / 'user_check_r1.csv'):
    if UC.get(r['id']) == '아님':
        pool['아님'].append(g_row(r, r['id'], '아님'))
    elif UC.get(r['id']) == '맞음' and r['stage'] == '봉오리':
        pool['봉오리'].append(g_row(r, r['id'], '봉오리'))
used = {r['gold_id'] for r in base if r['kind'] == 'gold'}
for v in pool.values():
    rng.shuffle(v); v.sort(key=lambda g: g['gold_id'] in used)
crop_cache = {}


def cell_img(c, size=None):
    size = size or a.cell; f = c['photo']; im = crop_cache.get(f)
    if im is None:
        crop_cache.clear(); im = crop_cache[f] = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB')
    b = xy(c); s = max(b[2] - b[0], b[3] - b[1]) * 1.8 + 24; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; x0, y0 = cx - s / 2, cy - s / 2; k = size / s
    t = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((size, size), Image.BICUBIC); ImageDraw.Draw(t).rectangle([(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k], outline=(0, 120, 255), width=2)
    return t


def b64(im, q=82):
    o = io.BytesIO(); im.save(o, 'JPEG', quality=q); return base64.b64encode(o.getvalue()).decode()


# 예시는 1차 파일의 것을 그대로 써서 기준을 같게 둔다
h0 = (VER / 'runs/team_check' / a.base / f'꽃판정_{a.base}_{REV[0]}.html').read_text(encoding='utf-8'); examples = json.loads(re.search(r'^const DATA=(.*);$', h0, flags=re.M)[1])['examples']
TPL = (VER / 'scripts/team_check_template.html').read_text(encoding='utf-8'); A0 = '<p>파란 박스 안의 대상이'; assert TPL.count(A0) == 1
TPL = TPL.replace(A0, '<p><b>추가 판정 파일입니다.</b> 1차에서 다른 조원이 본 칸입니다. 교차 확인을 위한 것이니 다른 사람의 판정을 묻거나 보지 말고 직접 판단해 주세요.</p>' + A0)
rows = []; seen = C.defaultdict(list)
for r in base:
    seen[r['reviewer']].append(r)
for r in REV:
    items = [dict(first[k], kind='candidate', gold='', gold_id='', first_reviewer=boxes[k][0]) for k in extra[r]]; n_g = round(a.gold * len(items)); gs = []
    for i in range(n_g):
        src = pool['아님'] if i % 2 else pool['봉오리']
        for g in src:
            if not any(c['photo'] == g['photo'] and max(ov(xy(c), xy(g))) >= 0.3 for c in items + seen[r] + gs):
                gs.append(dict(g, kind='gold', first_reviewer='')); src.remove(g); break
    items += gs; rng.shuffle(items); ids = [f'{r}{i + 1:03d}' for i in range(len(items))]
    S = Image.new('RGB', (a.cols * a.cell, ((len(items) + a.cols - 1) // a.cols) * a.cell), (40, 40, 40))
    for i in sorted(range(len(items)), key=lambda i: items[i]['photo']):
        S.paste(cell_img(items[i]), ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
    data = dict(reviewer=r, round=a.round, cols=a.cols, cell=a.cell, cells=ids, examples=examples); html = TPL.replace('__REVIEWER__', r + ' 추가').replace('__SHEET__', b64(S)).replace('__DATA__', json.dumps(data, ensure_ascii=False)); name = f'꽃판정_{a.round}_{r}.html'; (OUT / name).write_text(html, encoding='utf-8')
    for i, c in zip(ids, items):
        rows.append(dict(cell=i, reviewer=r, kind=c['kind'], photo=c['photo'], x0=c['x0'], y0=c['y0'], x1=c['x1'], y1=c['y1'], conf=c['conf'], gold=c['gold'], gold_id=c['gold_id'], first_reviewer=c['first_reviewer']))
    print(r, len(items), 'gold', len(gs), 'photos', len({c['photo'] for c in items}), name, round((OUT / name).stat().st_size / 1e6, 1), 'MB', flush=True)
write(L / f'team_check_{a.round}.csv', list(rows[0]), rows)
print('pairs', dict(sorted(pair.items())), 'load', dict(sorted(load.items())), flush=True)
print('MAKE_DONE total candidates', total, 'added', sum(len(v) for v in extra.values()), flush=True)
