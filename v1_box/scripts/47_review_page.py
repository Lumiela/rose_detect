# 사람 판정으로 바뀐 라벨을 눈으로 다시 확인하는 HTML을 만든다.
# review_changes.html: 변경 항목별 크롭과 판정 / review_photos.html: 사진 전체에 최종 라벨 표시 / checklist.csv: 번호 목록
# 기준은 판정 반영 전 커밋의 라벨이다. 사용: python 47_review_page.py [--base 3d8a17430] [--out runs/review_check]
import argparse, base64, collections as C, csv, html, io, json, subprocess
from common import *
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument('--base', default='3d8a17430'); ap.add_argument('--out', default=str(VER / 'runs' / 'review_check')); a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SPLITS = [('test', 'T', 'review_test_errors_r1.csv', 'review_test_fp.json', 'review_test_fn.json'),
          ('val', 'V', 'review_val_errors_r3.csv', 'review_val_fp.json', 'review_val_fn.json'),
          ('train', 'R', 'review_train_errors_r3.csv', 'review_train_fp.json', None)]
CATS = [('A', '추가한 라벨', '판정에 따라 새로 넣은 라벨입니다. 박스는 모델 예측입니다.'),
        ('D', '삭제한 라벨', '잎으로 판정해 지웠거나, 모델 박스로 교체한 기존 라벨입니다.'),
        ('I', '제외 영역', '알수없음, 일부만 보임 등으로 학습과 평가에서 뺀 영역입니다.'),
        ('S', '단계를 바꾼 라벨', '판정에 따라 단계를 고친 라벨입니다.'),
        ('P', '같은 꽃으로 보고 유지', '모델 박스(빨강)와 기존 라벨(파랑)을 같은 꽃으로 보고 기존 박스를 유지했습니다. 수치로만 판단한 항목입니다.'),
        ('L', '잎, 줄기로 판정', '모델이 꽃으로 잡았지만 잎이나 줄기로 판정해 라벨을 넣지 않았습니다.')]
COLOR = dict(keep=(40, 120, 255), add=(0, 220, 90), drop=(255, 140, 0), ign=(255, 0, 220), pred=(255, 30, 30))
NOOP = {'잎', '줄기'}
CODE = {'봉오리': 'B', '개화중': 'O', '만개': 'F'}


def old(name):
    t = subprocess.run(['git', 'show', f'{a.base}:v1_box/labels/{name}'], cwd=FLOWER, capture_output=True, check=True).stdout.decode('utf-8-sig')
    return list(csv.DictReader(io.StringIO(t)))


def ov(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    ap_, aq = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (ap_ + aq - i + 1e-9), i / min(ap_, aq)


man = {m['photo']: m for m in read(MANIFEST)}; cur, base, ign, ign0 = read(ANN), old('annotations.csv'), read(IGN), old('ignore_regions.csv')
ck = lambda r: (r['photo'], r['ann_id']); bk = lambda r: (r['photo'],) + tuple(int(float(r[c])) for c in ('x0', 'y0', 'x1', 'y1'))
cur_by, base_by = {ck(r): r for r in cur}, {ck(r): r for r in base}; ign_set, ign0_set = {bk(r) for r in ign}, {bk(r) for r in ign0}
items = []


def item(split, cat, photo, boxes, judge, ref, note=''):
    items.append(dict(split=split, cat=cat, photo=photo, boxes=boxes, judge=judge, ref=ref, note=note))


for split, pre, errs, jf, jn in SPLITS:
    E = read(VER / 'labels' / errs); JF = json.loads((VER / 'labels' / jf).read_text(encoding='utf-8')); JN = json.loads((VER / 'labels' / jn).read_text(encoding='utf-8')) if jn else {}
    fp = {r['id']: r for r in E if r['kind'] == 'fp'}; fn = {(r['photo'], r['ann_id']): r['id'] for r in E if r['kind'] == 'fn'}; in_split = lambda f: man[f]['split'] == split
    shown_ign = set()
    for r in cur:
        if in_split(r['photo']) and r['source'] == 'review':
            k = 'F' + r['ann_id'][1:]; item(split, 'A', r['photo'], [(box(r), 'add')], JF.get(k, ''), f"{k} → {r['ann_id']} {r['stage']}")
    for r in base:
        if not in_split(r['photo']) or ck(r) in cur_by:
            continue
        m = fn.get(ck(r)); j = JN.get(m, '') if m else ''
        if not j:  # 오탐 판정으로 지운 라벨
            c = [(sum(ov(box(r), box(e))), k) for k, e in fp.items() if e['photo'] == r['photo'] and ov(box(r), box(e))[1] >= 0.5]
            if c:
                m = max(c)[1]; j = JF.get(m, '')
        if bk(r) in ign_set:
            shown_ign.add(bk(r)); item(split, 'I', r['photo'], [(box(r), 'ign')], j, f"{m} ← {r['ann_id']} {r['stage']}", '기존 라벨을 제외 영역으로 옮김')
        else:
            near = [q for q in cur if q['photo'] == r['photo'] and q['source'] == 'review' and ov(box(r), box(q))[1] >= 0.5]
            item(split, 'D', r['photo'], [(box(r), 'drop')] + [(box(q), 'add') for q in near], j, f"{m} ← {r['ann_id']} {r['stage']}", '모델 박스로 교체' if near else '')
    for r in cur:
        o = base_by.get(ck(r))
        if in_split(r['photo']) and o and o['stage'] != r['stage']:
            m = fn.get(ck(r)); item(split, 'S', r['photo'], [(box(r), 'keep')], JN.get(m, ''), f"{m or ''} {r['ann_id']} {o['stage']} → {r['stage']}")
    for r in ign:
        if in_split(r['photo']) and bk(r) not in ign0_set and bk(r) not in shown_ign:
            c = [k for k, e in fp.items() if bk(e) == bk(r)]; shown_ign.add(bk(r))
            item(split, 'I', r['photo'], [(box(r), 'ign')], JF.get(c[0], '') if c else '', c[0] if c else '')
    added = {ck(r) for r in cur if r['source'] == 'review'}
    for k, e in fp.items():
        j = JF.get(k)
        if j is None:
            continue
        if j in NOOP:
            item(split, 'L', e['photo'], [(box(e), 'pred')], j, k); continue
        if (e['photo'], 'R' + k[1:]) in added or bk(e) in ign_set:
            continue
        c = [(sum(ov(box(e), box(q))), i) for i, q in enumerate(cur) if q['photo'] == e['photo'] and ov(box(e), box(q))[1] >= 0.3]
        if c:
            q = cur[max(c)[1]]; item(split, 'P', e['photo'], [(box(e), 'pred'), (box(q), 'keep')], j, f"{k} ~ {q['ann_id']} {q['stage']}")

# 번호를 매기고, 사진별로 한 번만 열어 크롭한다
num = C.Counter(); pre = {s[0]: s[1] for s in SPLITS}
order = {c[0]: i for i, c in enumerate(CATS)}
items.sort(key=lambda t: (['test', 'val', 'train'].index(t['split']), order[t['cat']], t['photo']))
for t in items:
    num[(t['split'], t['cat'])] += 1; t['id'] = f"{pre[t['split']]}{t['cat']}{num[(t['split'], t['cat'])]:03d}"
by = C.defaultdict(list)
for t in items:
    by[t['photo']].append(t)
THUMB = 190
for f, ts in by.items():
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB')
    for t in ts:
        bs = [b for b, _ in t['boxes']]; x0, y0, x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)
        s = max(x1 - x0, y1 - y0) * 1.8 + 60; ox, oy = (x0 + x1) / 2 - s / 2, (y0 + y1) / 2 - s / 2; z = THUMB / s
        c = im.crop((int(ox), int(oy), int(ox + s), int(oy + s))).resize((THUMB, THUMB), Image.BILINEAR); d = ImageDraw.Draw(c)
        for b, kind in t['boxes']:
            d.rectangle([(b[0] - ox) * z, (b[1] - oy) * z, (b[2] - ox) * z, (b[3] - oy) * z], outline=COLOR[kind], width=2)
        buf = io.BytesIO(); c.save(buf, 'JPEG', quality=82); t['img'] = base64.b64encode(buf.getvalue()).decode()

CSS = """:root{--bg:#fff;--fg:#1c1c1c;--mute:#666;--line:#ddd;--card:#f6f6f6}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mute:#aaa;--line:#3a3a3a;--card:#222}}
body{margin:0;padding:20px 16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}h1{font-size:22px;margin:0 0 8px}h2{font-size:18px;margin:28px 0 8px;border-top:1px solid var(--line);padding-top:16px}
p{margin:6px 0;color:var(--mute);max-width:70ch}table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid var(--line);padding:4px 10px;text-align:right}th:first-child,td:first-child{text-align:left}
summary{cursor:pointer;font-weight:600;padding:8px 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,190px));gap:10px}
.c{background:var(--card);border-radius:6px;overflow:hidden;font-size:12px}.c img{display:block;width:190px;height:190px}.c div{padding:6px 8px}.c b{font-size:13px}.m{color:var(--mute)}
.k{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}.p{margin:0 0 18px}.p img{max-width:100%;height:auto;display:block}"""
HEAD = lambda title: f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body>'
key = lambda k, t: f'<span class="k" style="background:rgb{COLOR[k]}"></span>{t}'
LEG = key('keep', '기존 라벨') + key('add', '추가한 라벨') + key('drop', '삭제한 라벨') + key('ign', '제외 영역') + key('pred', '모델 박스')
cnt = C.Counter((t['split'], t['cat']) for t in items); S = [s[0] for s in SPLITS]
h = [HEAD('라벨 변경 검수'), '<h1>라벨 변경 검수</h1><p>사용자 판정을 반영해 바뀐 라벨입니다. 틀린 항목은 번호로 알려 주세요. 예: <code>TA012 삭제</code>, <code>VI003 봉오리</code>, <code>RP010 개화중</code></p><p>번호 앞 글자는 분할(T test, V val, R train)이고, 둘째 글자는 아래 항목입니다.</p><p>', LEG, '</p><table><tr><th>항목</th>' + ''.join(f'<th>{s}</th>' for s in S) + '</tr>']
for c, name, _ in CATS:
    h.append(f'<tr><td>{c} {name}</td>' + ''.join(f'<td>{cnt[(s, c)]}</td>' for s in S) + '</tr>')
h.append('</table>')
for s in S:
    h.append(f'<h2>{s}</h2>')
    for c, name, desc in CATS:
        ts = [t for t in items if t['split'] == s and t['cat'] == c]
        if not ts:
            continue
        h.append(f"<details{' open' if c != 'L' else ''}><summary>{c} {name} ({len(ts)})</summary><p>{desc}</p><div class=grid>")
        for t in ts:
            h.append(f'<div class=c><img loading=lazy alt="{t["id"]}" src="data:image/jpeg;base64,{t["img"]}"><div><b>{t["id"]}</b> <span class=m>{t["photo"][9:15]}</span><br>{html.escape(t["judge"])}<br><span class=m>{html.escape(t["ref"])}{" · " + t["note"] if t["note"] else ""}</span></div></div>')
        h.append('</div></details>')
h.append('</body></html>'); (OUT / 'review_changes.html').write_text(''.join(h), encoding='utf-8')
with open(OUT / 'checklist.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['id', 'split', 'category', 'photo', 'judgement', 'ref', 'note'])
    for t in items:
        w.writerow([t['id'], t['split'], t['cat'], t['photo'], t['judge'], t['ref'], t['note']])

# 사진 전체: 최종 라벨
h = [HEAD('최종 라벨 사진'), '<h1>최종 라벨 사진</h1><p>판정 반영 후 라벨을 사진 전체에 그렸습니다. 글자는 단계입니다 (B 봉오리, O 개화중, F 만개).</p><p>', LEG.replace(key('pred', '모델 박스'), ''), '</p>']
LONG = 1400
for s in S:
    ph = sorted(m['photo'] for m in man.values() if m['split'] == s and m['label_status'] in COMPLETE)
    h.append(f'<h2>{s} ({len(ph)}장)</h2>')
    for f in ph:
        im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); z = LONG / max(im.size); im = im.resize((round(im.width * z), round(im.height * z)), Image.BILINEAR); d = ImageDraw.Draw(im)
        rect = lambda b, col: d.rectangle([v * z for v in b], outline=col, width=2)
        for r in ign:
            if r['photo'] == f:
                rect(box(r), COLOR['ign'])
        for r in base:
            if r['photo'] == f and ck(r) not in cur_by:
                rect(box(r), COLOR['drop'])
        n = 0
        for r in cur:
            if r['photo'] == f:
                col = COLOR['add'] if r['source'] == 'review' else COLOR['keep']; b = box(r); rect(b, col); d.text((b[0] * z + 2, b[1] * z + 1), CODE[r['stage']], fill=col); n += 1
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=72)
        h.append(f'<div class=p><details open><summary>{f} · 라벨 {n}개</summary><img loading=lazy alt="{f}" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"></details></div>')
h.append('</body></html>'); (OUT / 'review_photos.html').write_text(''.join(h), encoding='utf-8')
print(json.dumps({f'{s}/{c}': v for (s, c), v in sorted(cnt.items())}), len(items))
print((OUT / 'review_changes.html').stat().st_size, (OUT / 'review_photos.html').stat().st_size); print('DONE', flush=True)
