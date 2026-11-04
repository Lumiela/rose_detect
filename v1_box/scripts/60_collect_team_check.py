# 팀원 판정 파일을 모아 진행률, 기준 칸 일치, 겹친 칸의 일치, 판정 분포를 낸다. 라벨은 바꾸지 않는다. 판정이 갈린 칸은 runs/team_check/<회차>/disagree.csv로 남긴다.
# 사용: python 60_collect_team_check.py c1 [--dir 판정 파일 폴더, 기본 labels/team_check_<회차>]
import argparse, collections as C, glob, json
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('round'); ap.add_argument('--dir'); a = ap.parse_args()
D = Path(a.dir) if a.dir else VER / 'labels' / f'team_check_{a.round}'; rows = {r['cell']: r for r in read(VER / 'labels' / f'team_check_{a.round}.csv')}; res = {}
for p in sorted(glob.glob(str(D / '*.json'))):
    j = json.loads(Path(p).read_text(encoding='utf-8')); assert j['round'] == a.round, p; res[j['reviewer']] = j
per = C.defaultdict(dict); gold = C.defaultdict(C.Counter); dist = C.Counter()
for r, j in res.items():
    for k, v in j['verdicts'].items():
        c = rows[k]; assert c['reviewer'] == r, k
        if c['kind'] == 'gold':
            gold[r]['맞음' if v == c['gold'] else '다름'] += 1; gold[r][f"{c['gold']}->{v}"] += 1
        else:
            per[(c['photo'], c['x0'], c['y0'], c['x1'], c['y1'])][r] = v; dist[v] += 1
need = C.Counter(c['reviewer'] for c in rows.values())
for r in sorted(need):
    j = res.get(r); print(r, 'judged', j['judged'] if j else 0, '/', need[r], 'minutes', round(j['seconds'] / 60, 1) if j else '-', 'gold', dict(gold[r]))
two = [k for k, v in per.items() if len(v) >= 2]; agree = sum(1 for k in two if len(set(per[k].values())) == 1); print('candidates', len(per), 'verdicts', dict(dist), 'double judged', len(two), 'agree', agree)
OUT = VER / 'runs/team_check' / a.round; OUT.mkdir(parents=True, exist_ok=True); bad = [dict(photo=k[0], x0=k[1], y0=k[2], x1=k[3], y1=k[4], verdicts=json.dumps(per[k], ensure_ascii=False)) for k in two if len(set(per[k].values())) > 1]
if bad:
    write(OUT / 'disagree.csv', list(bad[0]), bad)
print('COLLECT_DONE', len(bad))
