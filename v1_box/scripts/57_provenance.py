# 라벨마다 박스를 만든 쪽과 사용자가 확인한 범위를 기록한다. labels/provenance.csv를 다시 만든다. 박스나 라벨 내용은 바꾸지 않는다.
# 확인 수준: 칸 판정(이 라벨이나 같은 꽃을 칸에서 판정), 사진 검수(test 17장 사진 단위 검수만), 없음.
# 사용: python 57_provenance.py
import collections as C, json
from common import *

L = VER / 'labels'; man = {m['photo']: m for m in read(MANIFEST)}; ann = read(ANN); masks = [json.loads(t) for t in (FLOWER / 'v2_seg/labels/masks.jsonl').read_text(encoding='utf-8').splitlines() if t.strip()]
ROUNDS = [('test1', 'review_test_errors_r1.csv', ['review_test_fp.json', 'review_test_fn.json']), ('val', 'review_val_errors_r3.csv', ['review_val_fp.json', 'review_val_fn.json']), ('train', 'review_train_errors_r3.csv', ['review_train_fp.json']), ('test2', 'review_test_errors_r2.csv', ['review_test_fp_r2.json', 'review_test_fn_r2.json']), ('test3', 'review_test_errors_r3.csv', ['review_test_fp_r3.json']), ('test4', 'review_test_errors_r4.csv', ['review_test_u_r4.json'])]
BOX = dict(L=('Claude', '꽃잎 색 후보를 스크립트로 뽑고 Claude가 썸네일로 판정 (2026-10-30)'), D=('Claude', 'Claude가 격자 이미지를 보고 작성'), R=('모델', '§det/first§ 예측을 사용자 판정으로 추가'), P=('모델', '§det/second§ 예측을 임시 라벨 표본 판정으로 추가'), T=('모델', '§det/third_pseudo§ 예측을 test 2차 판정으로 추가'), V=('모델', '§seg/first§ 예측을 test 3차 판정으로 추가'), X=('모델', '§det/second§ 예측, 신뢰도 0.7 이상 (임시 라벨)'))


def ov(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h; x, y = (p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])
    return i / (x + y - i + 1e-9), i / (min(x, y) + 1e-9)


byp = C.defaultdict(list); checks = C.defaultdict(list)
for r in ann:
    byp[r['photo']].append(r)
for name, csv_, js in ROUNDS:
    E = {r['id']: r for r in read(L / csv_)}
    for j in js:
        for k, v in json.loads((L / j).read_text(encoding='utf-8')).items():
            e = E[k]
            if e['ann_id']:
                checks[(e['photo'], e['ann_id'])].append(f'{name}:{k}:{v}')
            else:  # 오탐 칸은 라벨 번호가 없어 겹침으로 같은 꽃을 찾는다
                b = box(e); best, bv = None, 0
                for r in byp[e['photo']]:
                    io, os_ = ov(b, box(r))
                    if (io >= 0.3 or os_ >= 0.6) and io + os_ > bv:
                        best, bv = r, io + os_
                if best is not None:
                    checks[(e['photo'], best['ann_id'])].append(f'{name}:{k}:{v}')
rows = []
for r in ann:
    a = r['ann_id']; k = a[0]; key = (r['photo'], a); ch = checks.get(key, []); s = man[r['photo']]['split']
    lvl = '칸 판정' if ch or k in 'RPTV' else '사진 검수' if s == 'test' else '없음'
    rows.append(dict(photo=r['photo'], split=s, key=a, stage=r['stage'], source=r['source'], box_by=BOX[k][0], box_how=BOX[k][1].replace('§', ''), user_check=lvl, detail=' | '.join(ch), mask='', x0=r['x0'], y0=r['y0'], x1=r['x1'], y1=r['y1']))
mk = {(m['photo'], m['key']) for m in masks}
for x in rows:
    x['mask'] = 'SAM2.1 초안, 사람 확인 없음' if (x['photo'], x['key']) in mk else ''
for m in masks:
    if m['source'] == 'pseudo':
        b = m['box']; rows.append(dict(photo=m['photo'], split=m['split'], key=m['key'], stage='', source='pseudo', box_by=BOX['X'][0], box_how=BOX['X'][1].replace('§', ''), user_check='없음', detail='', mask='SAM2.1 초안, 사람 확인 없음', x0=b[0], y0=b[1], x1=b[2], y1=b[3]))
write(L / 'provenance.csv', list(rows[0]), rows)
print(len(rows), sorted(C.Counter((r['split'], r['source'], r['user_check']) for r in rows).items()), C.Counter((r['split'], r['user_check']) for r in rows))
