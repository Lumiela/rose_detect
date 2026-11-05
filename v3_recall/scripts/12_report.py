# v3 리포트를 만든다: REPORT.html(그림 포함), REPORT.md(요약). 수치는 평가 파일과 최종 처리 안에서 읽는다. 기준 칸 수치(GOLD)만 로컬 비교 결과에서 옮겼다.
# 사용: python 12_report.py A_base
import base64, collections as C, html, re, sys
from decimal import Decimal, ROUND_HALF_UP
from common import *
from PIL import Image, ImageDraw, ImageOps

NAME = sys.argv[1] if len(sys.argv) > 1 else 'A_base'; J = lambda p: json.loads(Path(p).read_text(encoding='utf-8')); RS = {'v1': V1 / 'runs/det/third_pseudo', 'v2': FLOWER / 'v2_seg/runs/seg/first', 'v3': VER / 'runs/seg' / NAME}
GOLD = dict(team=(80, 103, 4), claude=(80, 103, 10))  # 맞음, 칸, 봉오리와 아님 반대. human_label/flower_compare/요약.txt
M = {k: J(r / 'eval/photo_metrics.json') for k, r in RS.items()}; E = {k: J(r / 'eval/e2e_metrics.json') for k, r in RS.items()}; I3 = {k: J(r / 'eval_iou03/photo_metrics.json') for k, r in RS.items()}; CM = J(RS['v3'] / 'eval/compare_v1_v2.json'); V = {n: J(VER / 'runs/seg' / n / 'eval_val/photo_metrics.json')['val']['chosen'] for n in ('A_base', 'C_mask2', 'B_960')}; V['v2'] = M['v2']['val']['chosen']
plan = read(FLOWER / 'human_label/최종처리_안.csv'); added = read(VER / 'labels/team_added.csv'); skip = read(VER / 'labels/team_skipped.csv'); ann3, ann1 = read(ANN), read(V1 / 'labels/annotations.csv'); man = {m['photo']: m for m in read(MANIFEST)}; masks = read_masks(); prov = read(VER / 'labels/provenance.csv')
FIG = VER / 'report'; FIG.mkdir(exist_ok=True); f3 = lambda v: str(Decimal(str(v)).quantize(Decimal('0.001'), ROUND_HALF_UP)); pc = lambda a, b: f'{a / b * 100:.1f}%'; ci = lambda a: f'{a[0]:+.3f} ~ {a[1]:+.3f}'
GT, IG = C.defaultdict(list), C.defaultdict(list)
for r in ann3:
    GT[r['photo']].append(r)
for r in read(IGN):
    IG[r['photo']].append(box(r))
P = {k: J(RS[k] / 'eval/preds_test.json') for k in ('v2', 'v3')}; CONF = CM['conf']


def draw(k, key, width=800):
    f = next(x for x in P[k] if key in x); im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); z = width / im.width; im = im.resize((width, round(im.height * z)), Image.BILINEAR); d = ImageDraw.Draw(im); G = [box(r) for r in GT[f]]; used = set()
    for p in sorted((q for q in P[k][f] if q[4] >= CONF[k]), key=lambda q: -q[4]):
        v = [ov(p[:4], g)[0] if j not in used else 0 for j, g in enumerate(G)]; j = max(range(len(v)), key=v.__getitem__) if v else -1; hit = j >= 0 and v[j] >= 0.5
        if hit:
            used.add(j)
        elif any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in IG[f]):
            continue
        if p[5]:
            q = [(x * z, y * z) for x, y in p[5]]; d.line(q + q[:1], fill=(40, 230, 80) if hit else (255, 40, 40), width=2)
    for j, g in enumerate(G):
        if j not in used:
            d.rectangle([g[0] * z - 2, g[1] * z - 2, g[2] * z + 2, g[3] * z + 2], outline=(60, 140, 255), width=2)
    out = FIG / f'cmp_{key}_{k}.jpg'; im.save(out, quality=82); return out.name


FIGS = [(key, draw('v2', key), draw('v3', key)) for key in ('140440', '135048', '134905')]
tiers = C.Counter(r['등급'] for r in plan); same = sum(r['판정1'] == r['판정2'] for r in plan); fin = C.Counter(r['처리'] for r in plan); new = [r for r in ann3 if r['ann_id'][0] == 'H']; sz = C.Counter(); [sz.update(['<40' if s < 40 else '40-60' if s < 60 else '60-80' if s < 80 else '80-160' if s < 160 else '>=160']) for s in (max(box(r)[2] - box(r)[0], box(r)[3] - box(r)[1]) for r in new)]
tr1 = sum(man[r['photo']]['split'] == 'train' for r in ann1); tr3 = sum(man[r['photo']]['split'] == 'train' for r in ann3); pt = [r for r in prov if r['split'] == 'train']; chk = C.Counter(r['user_check'] for r in pt); nm = [m for m in masks if m['key'][0] == 'H']; mi = lambda s: C.Counter('fail' if (not m.get('poly') or m.get('use') is False) else 'hi' if m['box_iou'] >= 0.7 else 'mid' for m in s)
ci1, ci2 = CM['rows']['v1'], CM['rows']['v2']; t = {k: M[k]['test']['chosen'] for k in RS}; e = {k: E[k]['test'] for k in RS}; by = lambda k, x: f3(t[k]['recall_by'][x][0])
blocks = []
H = lambda s: blocks.append(('h', s)); PG = lambda s: blocks.append(('p', s)); UL = lambda a: blocks.append(('ul', a)); T = lambda h, b, n=(): blocks.append(('t', h, b, n)); N = lambda s: blocks.append(('note', s))
H('요약')
UL([f'**라벨을 고쳐 재학습했습니다.** 조원 5명의 교차 판정과 사용자 판단으로 train 라벨 {len(new):,}개를 추가하고, 구조와 학습 설정은 v2와 같게 두었습니다.', f"**test 재현율이 {f3(t['v2']['recall'])}에서 {f3(t['v3']['recall'])}로 올랐습니다.** 긴 변 80px 미만은 {by('v2', '<80px')}에서 {by('v3', '<80px')}이고, 사진당 봉오리 개수 오차는 {e['v2']['count_mae_per_photo'][0]:.2f}에서 {e['v3']['count_mae_per_photo'][0]:.2f}입니다.", f"**오탐이 늘었습니다.** 정밀도는 {f3(t['v2']['precision'])}에서 {f3(t['v3']['precision'])}이고, 오탐은 {t['v2']['fp']}개에서 {t['v3']['fp']}개입니다. 이 오탐은 사람이 판정하지 않았습니다.", '**라벨 효과라고 단정하지 않습니다.** 조건마다 한 번씩만 학습했고 임계값도 0.65에서 0.60으로 달라졌습니다.', '**ESP 카메라에서는 타일을 880으로 맞췄습니다.** 사진 6장의 봉오리 3개를 16번 중 4번에서 9번 찾게 됐고, 재학습은 하지 않았습니다. 표본이 작아 임시 값입니다.', '**최종 모델은 아직 정하지 않았습니다.**'])
H('데이터')
T(['항목', 'v2', 'v3'], [['train 라벨 원본', f'{tr1:,}', f'{tr3:,}'], ['train 전체 (임시 라벨 포함)', '4,633', f'{len(pt):,}'], ['사람이 확인한 train', '12.0%', pc(len(pt) - chk['없음'], len(pt))], ['학습 대상 수 (타일 기준)', '12,394', '15,872'], ['val, test 라벨', '-', 'v1과 동일 (스크립트로 확인)']], (1, 2))
T(['새 라벨 크기', '개수'], [[k, sz[k]] for k in ('<40', '40-60', '60-80', '80-160', '>=160') if sz[k]], (1,))
UL([f'**라벨 사본:** v1_box/labels를 복사해 고쳤습니다. v1_box/labels는 바뀌지 않았습니다.', f"**표본 확인 13칸:** 라벨 8개를 지우고, 임시 라벨 5개와 함께 제외 영역이나 차단 목록으로 옮겼습니다.", f"**겹침 조정:** {len(skip)}칸을 뺐습니다 (labels/team_skipped.csv).", f"**마스크:** 새 라벨 {len(nm):,}개를 SAM2.1로 만들었습니다. 박스와의 IoU 0.7 이상 {pc(mi(nm)['hi'], len(nm))}, 실패 {mi(nm)['fail']}칸입니다. 사람이 그린 정답이 없어 마스크 정확도는 평가하지 못했습니다."])
H('사람 판정')
T(['판정', '내용'], [['후보', f'train에서 third_pseudo가 신뢰도 0.5 이상으로 찾았지만 라벨이 없는 {len(plan):,}칸'], ['조원', f'칸마다 2명, 10개 짝이 170~172칸씩 겹침. 같은 판정 {same:,}칸 ({pc(same, len(plan))})'], ['Claude', '조원 판정을 읽지 않은 별도 에이전트가 판정, 비교 전에 결과 고정'], ['사용자', '갈린 390칸, 등급 표본 130칸, 반복 15칸을 판정을 가리고 판단']])
T(['등급', '칸', '처리'], [['높음 (사람 2명과 Claude 같음)', tiers['높음'], '판정대로'], ['보통 (Claude만 다름)', tiers['보통'], '판정대로, 표본 확인'], ['판단 필요 (사람 2명이 다름)', tiers['판단 필요'], '사용자 판단'], ['제외 (둘 다 알수없음)', tiers['제외'], '제외 영역']], (1,))
UL(['**등급 검증:** 높음 표본에서 봉오리 30칸 중 30칸, 아님 15칸 중 14칸이 사용자와 같습니다. 보통 표본 50칸에서 사용자는 사람과 38칸, Claude와 8칸 같았습니다.', f"**기준 칸 103칸:** 조원 {GOLD['team'][0]}칸, Claude {GOLD['claude'][0]}칸이 사용자 판정과 같고, 봉오리와 아님의 반대는 조원 {GOLD['team'][2]}칸, Claude {GOLD['claude'][2]}칸입니다.", f"**처리:** 라벨 추가 {fin['라벨 추가']:,}, 제외 영역 {fin['제외 영역']}, 차단 {fin['차단']}칸. 상세 비교는 human_label의 비교 리포트에 있습니다."])
H('학습 조건 선택 (val)')
T(['조건', '임계값', '정밀도', '재현율', 'F1', '오탐/놓침', '80px 미만'], [[n, V[k]['conf'], f3(V[k]['precision']), f3(V[k]['recall']), f3(V[k]['f1']), f"{V[k]['fp']}/{V[k]['fn']}", f3(V[k]['recall_by']['<80px'][0])] for n, k in (('v2 seg/first', 'v2'), ('A 기준 (라벨만 변경)', 'A_base'), ('C mask_ratio 2', 'C_mask2'), ('B 입력 960', 'B_960'))], (1, 2, 3, 4, 5, 6))
PG('목표인 재현율과 F1이 가장 높은 A를 val로 골랐고, test는 A만 한 번 평가했습니다. val에는 노란 꽃 구간이 없고 사진이 8장이라 조건 차이가 우연일 수 있습니다. C와 B의 test는 보지 않았습니다.')
H('test 비교')
T(['항목', 'v1', 'v2', 'v3'], [['임계값 (val 기준)'] + [CONF[k] for k in RS], ['정밀도'] + [f3(t[k]['precision']) for k in RS], ['재현율'] + [f3(t[k]['recall']) for k in RS], ['F1 (IoU 0.5)'] + [f3(t[k]['f1']) for k in RS], ['F1 (IoU 0.3)'] + [f3(I3[k]['test']['chosen']['f1']) for k in RS], ['오탐 / 놓침'] + [f"{t[k]['fp']} / {t[k]['fn']}" for k in RS], ['사진당 봉오리 개수 오차'] + [f"{e[k]['count_mae_per_photo'][0]:.2f}" for k in RS], ['봉오리 수 (라벨 473)'] + [e[k]['count_total_pred'][0] for k in RS], ['매크로 F1 (단계까지)'] + [f3(e[k]['macro_f1']) for k in RS]], (1, 2, 3))
T(['v3와의 차이 95% 구간', 'v3 - v1', 'v3 - v2'], [['정밀도', ci(ci1['IoU.5']['P']), ci(ci2['IoU.5']['P'])], ['재현율', ci(ci1['IoU.5']['R']), ci(ci2['IoU.5']['R'])], ['F1 (IoU 0.5)', ci(ci1['IoU.5']['F1']), ci(ci2['IoU.5']['F1'])], ['F1 (IoU 0.3)', ci(ci1['IoU.3']['F1']), ci(ci2['IoU.3']['F1'])], ['F1 (IoU 0.5, 추가 라벨 제외)', ci(ci1['제외 IoU.5']['F1']), ci(ci2['제외 IoU.5']['F1'])], ['봉오리 개수 오차', f"{ci1['개수 오차']['lo'][0]:+.2f} ~ {ci1['개수 오차']['hi'][0]:+.2f}", f"{ci2['개수 오차']['lo'][0]:+.2f} ~ {ci2['개수 오차']['hi'][0]:+.2f}"]], (1, 2))
UL(['**확실한 차이:** 재현율은 v1, v2보다 높고, 정밀도는 v2보다 낮으며, 봉오리 개수 오차는 v2보다 낮습니다.', '**확실하지 않은 차이:** F1은 v1, v2와 구간이 0을 포함합니다. 봉오리 개수 오차도 v1과는 구간이 0을 포함합니다.', '**구간:** test 17장을 복원 추출해 4,000번 계산했습니다. 사진 수가 적어 대략적인 값입니다.'])
T(['재현율', '라벨', 'v1', 'v2', 'v3'], [[n, t['v3']['recall_by'][k][1], by('v1', k), by('v2', k), by('v3', k)] for n, k in (('긴 변 80px 미만', '<80px'), ('80~160px', '80-160px'), ('160px 이상', '>=160px'), ('봉오리', '봉오리'), ('개화중', '개화중'), ('만개', '만개'), ('빨간 꽃 구간', 'test_red'), ('노란 꽃 구간', 'test_yellow'), ('혼합 구간', 'test_mixed'))], (1, 2, 3, 4))
blocks.append(('f', FIGS))
H('ESP 카메라 추론')
PG('휴대폰 test 점수는 큐브 카메라 성능이 아니라서, 새 ESP 카메라 사진 6장(2560×1920)에 꽃 모델을 돌렸습니다. 같은 트레이를 시점만 바꿔 찍은 사진입니다. 사람이 확인한 봉오리는 3개이고, B는 위치를 믿을 수 있는 4장만 세어 모두 16번입니다. 임계값은 휴대폰 val 값입니다.')
blocks.append(('i', 'esp_buds_abc.jpg', 'ESP 사진의 봉오리 A, B, C'))
T(['봉오리 크기', '휴대폰', 'ESP'], [['사진', '3000×4000', '2560×1920'], ['긴 변 중앙값', 'train 68px, test 80px', '약 55px'], ['입력 안 (타일 1280)', '34~40px', '약 27px']], (1, 2))
SC = J(VER / 'runs/esp_compare/low/scale.json'); VIEW = {1280: 7, 1100: 7, 960: 13, 880: 13, 819: 13, 720: 21, 640: 21}
T(['v3 설정', '입력 안의 봉오리', 'A', 'B', 'C', '합계 / 16', '다른 예측', '모델 입력 수'], [[r['name'].replace('tile', '타일 ').replace('확대', '확대 ').replace('+타일 ', ' + 타일 '), f"약 {55 / (r['tile'] / r['zoom']) * 640:.0f}px", r['A'], r['B'], r['C'], r['A'] + r['B'] + r['C'], r['other'], VIEW[r['tile']] if r['zoom'] == 1 else '13'] for r in SC], (1, 2, 3, 4, 5, 6, 7))
T(['조건 (16번 중)', '타일 1280', '타일 960'], [['v3 원본', 4, 8], ['v3 + 화이트밸런스', 3, 8], ['입력 960 모델 B_960', 2, 7], ['B_960 + 화이트밸런스', 1, 4], ['v1 박스', 4, 6], ['v2 세그', 1, 3]], (1, 2))
UL(['**축척:** 입력 크기는 학습과 같은 640이고, 봉오리가 입력에서 작아지는 것이 문제였습니다. 재학습 없이 타일만 줄여 v3가 찾은 횟수가 4번에서 9번으로 늘었습니다. 사진을 1.45배 키워도 결과가 같아, 타일을 바꾸는 쪽을 택했습니다.', '**설정:** cameras.json에 카메라별 타일을 두었습니다. phone은 1280, esp는 880이고, 40_flower_ids.py의 --camera로 고릅니다. 휴대폰 설정의 예측은 기존 평가와 같은 것을 사진 1장으로 확인했습니다.', '**타일 880의 다른 예측 1개:** 사진 b676fe의 화분 위 빨간 표시등 반사로 보입니다 (Claude의 인상).', '**화이트밸런스:** 잎 모델의 gray_world는 사진 평균 G를 2~5 낮출 뿐 효과가 없어 넣지 않았습니다. 다른 LED 상태는 시험하지 못했습니다.', '**봉오리 C:** 어떤 조건에서도 찾지 못했습니다 (신뢰도 최고 0.32). ESP 사진을 학습에 넣어야 풀릴 것으로 보지만 시험하지 않았습니다.', '**기존 큐브 사진 69장:** 흐린 다른 카메라 사진입니다. v3가 1,115개를 찾고 개화중이 45.7%로 나왔습니다. 휴대폰 test의 약 7%와 크게 달라 단계 판정을 믿기 어렵습니다. 라벨은 없습니다.'])
N('타일 880은 사진 6장, 봉오리 3개로 고른 임시 값입니다. 고른 사진으로 잰 값이라 평가 점수가 아닙니다. 다른 날 찍은 ESP 사진으로 다시 정해야 합니다.')
H('한계')
UL(['**학습은 조건마다 한 번입니다.** 시드를 바꾼 반복이 없어 라벨 효과와 학습 운을 나누지 못했습니다.', '**임계값이 다릅니다.** v2는 0.65, v3는 0.60입니다. 각자 val F1로 정한 값입니다. 재현율 상승과 오탐 증가에 이 차이도 들어 있습니다.', '**test 오탐 53개는 판정하지 않았습니다.** 이전 회차처럼 빠진 봉오리가 섞여 있으면 실제 정밀도는 더 높을 수 있습니다.', '**test를 한 번 봤습니다.** 이 결과로 임계값이나 조건을 바꾸면 안 됩니다.', '**test 라벨은 v1, v2의 오류를 보며 고쳤습니다.** 추가 라벨 44개를 뺀 값도 함께 적었습니다.', '**새 라벨 박스는 모델 예측입니다.** 사람은 칸을 판정했을 뿐 박스를 고치지 않았습니다. 학습에 쓰던 사진의 예측이라 그 모델의 습관이 남을 수 있습니다.', '**모델이 찾은 칸만 판정했습니다.** 모델과 라벨이 함께 놓친 꽃은 여전히 알 수 없습니다.', '**단계 분류기는 재학습하지 않았습니다.** 개화중이 여전히 가장 약합니다.', '**test는 휴대폰 사진 기준입니다.** ESP 카메라에서는 라벨이 있는 평가가 없고, 위의 봉오리 3개로 방향만 봤습니다. Jetson 추론 시간도 재지 않았습니다.'])
H('다음 단계')
UL(['다른 날, 다른 화분, LED 상태별로 ESP 사진을 찍어 교차 판정으로 test를 만들고, 타일과 임계값을 다시 정합니다.', 'ESP 라벨로 v3를 파인튜닝해 봉오리 C 같은 경우를 봅니다.', 'A를 시드를 바꿔 다시 학습해 우연의 크기를 잽니다.', '쓸 모델을 정합니다. 마스크가 필요 없다면 v1도 후보입니다.', '개화중 기준을 다시 맞추고 분류기를 ESP 칸과 흐림 증강으로 재학습합니다.'])
CSS = (V1 / 'report/style.css').read_text(encoding='utf-8')


def inline(s):
    s = html.escape(str(s), quote=False); return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)


def uri(n):
    return 'data:image/jpeg;base64,' + base64.b64encode((FIG / n).read_bytes()).decode()


o = []; sec = 0; op = False; md = [f'# 꽃 모델 v3 리포트: 팀 교차 판정 라벨', '', '> 그림을 포함한 리포트: [REPORT.html](REPORT.html)', '']
for b in blocks:
    if b[0] == 'h':
        o.append('</section>' if op else ''); sec += 1; o.append(f'<section><h2><span class="num">{sec:02d}</span>{b[1]}</h2>'); op = True; md += [f'## {b[1]}', '']
    elif b[0] in ('p', 'note'):
        o.append(f'<p>{inline(b[1])}</p>'); md += [b[1], '']
    elif b[0] == 'ul':
        o.append('<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in b[1]) + '</ul>'); md += [f'- {x}' for x in b[1]] + ['']
    elif b[0] == 'i':
        o.append(f'<figure><img alt="{b[2]}" src="{uri(b[1])}"><figcaption>{b[2]}</figcaption></figure>'); md += [f'![{b[2]}](report/{b[1]})', '']
    elif b[0] == 't':
        cn = lambda i: ' class="n"' if i in b[3] else ''; o.append('<div class="tbl"><table><thead><tr>' + ''.join(f'<th{cn(i)}>{x}</th>' for i, x in enumerate(b[1])) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td{cn(i)}>{inline(x)}</td>' for i, x in enumerate(r)) + '</tr>' for r in b[2]) + '</tbody></table></div>')
        md += ['| ' + ' | '.join(map(str, b[1])) + ' |', '|' + '|'.join('---:' if i in b[3] else '---' for i in range(len(b[1]))) + '|'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in b[2]] + ['']
    else:
        o.append('<h3>같은 사진 (초록 맞게 찾음, 빨강 오탐, 파랑 놓침)</h3><div class="figs">')
        for key, a, c in b[1]:
            o.append(f'<figure><img alt="v2 {key}" src="{uri(a)}"><figcaption>v2, 사진 {key}</figcaption></figure><figure><img alt="v3 {key}" src="{uri(c)}"><figcaption>v3, 사진 {key}</figcaption></figure>'); md += [f'![v2 {key}](report/{a})', f'![v3 {key}](report/{c})', '']
        o.append('</div>')
o.append('</section>')
K = [(f3(t['v3']['recall']), f"test 재현율 (v2 {f3(t['v2']['recall'])})"), (by('v3', '<80px'), f"80px 미만 재현율 (v2 {by('v2', '<80px')})"), (f3(t['v3']['precision']), f"test 정밀도 (v2 {f3(t['v2']['precision'])})"), (f"{e['v3']['count_mae_per_photo'][0]:.2f}", f"봉오리 개수 오차 (v2 {e['v2']['count_mae_per_photo'][0]:.2f})"), (f'{len(new):,}', '추가한 train 라벨')]
page = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 v3 리포트</title><style>{CSS}.wrap{{grid-template-columns:minmax(0,1fr)}}</style></head><body><div class="wrap"><header><div class="eyebrow">FLOWER MODEL · v3_recall · {NAME}</div><h1>꽃 모델 v3: 팀 교차 판정 라벨</h1><div class="meta"><span>작업일 2026-11-05</span><span>비교 기준 v1_box, v2_seg</span><span>최종 확정 전</span></div><div class="kpis">' + ''.join(f'<div class="kpi"><b>{a}</b><span>{c}</span></div>' for a, c in K) + '</div></header>' + ''.join(o) + '<footer>수치는 평가 결과 파일에서 12_report.py로 읽었습니다. 휴대폰 사진 기준입니다.</footer></div></body></html>'
(VER / 'REPORT.html').write_text(page, encoding='utf-8'); (VER / 'REPORT.md').write_text('\n'.join(md) + '\n', encoding='utf-8'); print('REPORT_DONE', len(page) // 1024, 'KB', K, FIGS)
