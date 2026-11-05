# v3 라벨마다 박스를 만든 쪽과 사람이 확인한 범위를 기록한다. 기존 라벨과 임시 라벨은 v1 기록을 이어받고, 새 라벨은 칸별 판정을 적는다. labels/provenance.csv를 다시 만든다.
# 확인 수준: 사용자 판정, 팀 2인 판정, 칸 판정(v1), 사진 검수(v1), 없음.
# 사용: python 02_provenance.py
import collections as C
from common import *

L = VER / 'labels'; man = {m['photo']: m for m in read(MANIFEST)}; old = {(r['photo'], r['key']): r for r in read(V1 / 'labels/provenance.csv')}; team = {r['ann_id']: r for r in read(L / 'team_added.csv')}; masks = {(m['photo'], m['key']): m for m in read_masks()}; rows = []
F = ['photo', 'split', 'key', 'stage', 'source', 'box_by', 'box_how', 'user_check', 'detail', 'mask', 'x0', 'y0', 'x1', 'y1']; M = 'SAM2.1 초안, 사람 확인 없음'
for r in read(ANN):
    k = (r['photo'], r['ann_id']); hasm = bool(masks.get(k, {}).get('poly'))
    if r['ann_id'][0] == 'H':
        t = team[r['ann_id']]; rows.append(dict(photo=r['photo'], split=man[r['photo']]['split'], key=r['ann_id'], stage=r['stage'], source='team', box_by='모델', box_how='det/third_pseudo 예측을 팀 교차 판정으로 추가', user_check='사용자 판정' if t['사용자'] else '팀 2인 판정', detail=f"{t['판정자1']}:{t['판정1']} | {t['판정자2']}:{t['판정2']} | Claude:{t['Claude']}" + (f" | 사용자:{t['사용자']}" if t['사용자'] else '') + f" | 등급:{t['등급']}", mask=M if hasm else '', x0=r['x0'], y0=r['y0'], x1=r['x1'], y1=r['y1']))
    else:
        o = old[k]; rows.append({**{c: o[c] for c in F}, 'mask': M if hasm else ''})
for k, m in masks.items():
    if m['source'] == 'pseudo':
        o = old.get(k); b = m['box']; rows.append(dict(photo=m['photo'], split=m['split'], key=m['key'], stage='', source='pseudo', box_by='모델', box_how='det/second 예측, 신뢰도 0.7 이상 (임시 라벨)', user_check=o['user_check'] if o else '없음', detail=o['detail'] if o else '', mask=M if m.get('poly') else '', x0=b[0], y0=b[1], x1=b[2], y1=b[3]))
write(L / 'provenance.csv', F, rows); print(len(rows), sorted(C.Counter((r['split'], r['user_check']) for r in rows).items())); print(sorted(C.Counter((r['split'], r['source']) for r in rows).items()))
