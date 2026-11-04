# v1 third_pseudo와 v2 seg/first를 같은 라벨로 비교해 COMPARE.html, COMPARE.md, runs/seg/first/compare/v1_v2.json을 만든다. 본문의 §는 코드 표시다.
# 표의 값은 평가 결과 파일에서 읽는다. 차이 구간은 test 사진을 복원 추출한 v1-v2의 95% 구간이고, 저장된 예측으로 계산한다. 먼저 34_time_models.py를 실행한다.
# 사용: python 33_compare_page.py
import base64, collections as C, re
from decimal import Decimal, ROUND_HALF_UP
from common import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps

J = lambda p: json.loads(Path(p).read_text(encoding='utf-8')); R1 = V1 / 'runs/det/third_pseudo'; R2 = VER / 'runs/seg/first'; FIG = VER / 'report'; OUT = R2 / 'compare'
M1, M2 = J(R1 / 'eval/photo_metrics.json'), J(R2 / 'eval/photo_metrics.json'); E1, E2 = J(R1 / 'eval/e2e_metrics.json'), J(R2 / 'eval/e2e_metrics.json'); I1, I2 = J(R1 / 'eval_iou03/photo_metrics.json'), J(R2 / 'eval_iou03/photo_metrics.json'); TM = J(OUT / 'timing.json')
P1, P2 = J(R1 / 'eval/preds_test.json'), J(R2 / 'eval/preds_test.json'); assert set(P1) == set(P2); PH = sorted(P1); CONF = M1['val']['chosen']['conf']; assert CONF == M2['val']['chosen']['conf']
GT, IG = C.defaultdict(list), C.defaultdict(list)
for r in read(ANN):
    GT[r['photo']].append(r)
for r in read(IGN):
    IG[r['photo']].append(box(r))
f3 = lambda v: str(Decimal(str(v)).quantize(Decimal('0.001'), ROUND_HALF_UP)); f2 = lambda v: f'{v:.2f}'; rng_txt = lambda a, b: f'{a:.3f} ~ {b:.3f}'


def match(f, P, iou=0.5, neutral=False):
    # (맞은 예측, 오탐, 놓친 라벨 번호). neutral이면 판정으로 추가한 라벨(T, V)을 제외 영역처럼 다룬다
    rows = [r for r in GT[f] if not (neutral and r['ann_id'][0] in 'TV')]; G = [box(r) for r in rows]; I_ = IG[f] + ([box(r) for r in GT[f] if r['ann_id'][0] in 'TV'] if neutral else []); used = set(); tp, fp = [], []
    for p in sorted((q for q in P[f] if q[4] >= CONF), key=lambda q: -q[4]):
        v = [ov(p[:4], g)[0] if k not in used else 0 for k, g in enumerate(G)]; k = max(range(len(v)), key=v.__getitem__) if v else -1
        if k >= 0 and v[k] >= iou:
            used.add(k); tp.append(p)
        elif not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in I_):
            fp.append(p)
    return tp, fp, [rows[k] for k in range(len(rows)) if k not in used]


def prf(c):
    tp, fp, fn = c.sum(0); p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return np.array([p, r, 2 * p * r / max(p + r, 1e-9)])


def table_counts(P, iou, neutral):
    return np.array([[len(x) for x in match(f, P, iou, neutral)] for f in PH], float)


def diff_ci(iou, neutral):
    A, B = table_counts(P1, iou, neutral), table_counts(P2, iou, neutral); g = np.random.default_rng(0); D = []
    for _ in range(4000):
        k = g.integers(len(PH), size=len(PH)); D.append(prf(A[k]) - prf(B[k]))
    return prf(A), prf(B), np.percentile(D, [2.5, 97.5], axis=0)


def verdict(lo, hi, good='높음'):
    return '**v1 우세**' if lo > 0 else '**v2 우세**' if hi < 0 else '확실하지 않음'


a5, b5, c5 = diff_ci(0.5, False); a3, b3, c3 = diff_ci(0.3, False); an5, bn5, cn5 = diff_ci(0.5, True); an3, bn3, cn3 = diff_ci(0.3, True); t1, t2 = M1['test']['chosen'], M2['test']['chosen']; v1, v2 = M1['val']['chosen'], M2['val']['chosen']; e1, e2 = E1['test'], E2['test']; ve1, ve2 = E1['val'], E2['val']
p1 = {c['photo']: c for c in e1['per_photo']}; p2 = {c['photo']: c for c in e2['per_photo']}; cnt1 = np.array([[abs(x - y) for x, y in zip(p1[f]['label'], p1[f]['pred'])] for f in PH], float); cnt2 = np.array([[abs(x - y) for x, y in zip(p2[f]['label'], p2[f]['pred'])] for f in PH], float)
g = np.random.default_rng(0); DC = [cnt1[k].mean(0) - cnt2[k].mean(0) for k in (g.integers(len(PH), size=len(PH)) for _ in range(4000))]; cc = np.percentile(DC, [2.5, 97.5], axis=0)
better = ((cnt1[:, 0] < cnt2[:, 0]).sum(), (cnt1[:, 0] > cnt2[:, 0]).sum(), (cnt1[:, 0] == cnt2[:, 0]).sum())
T_TEST = (['항목', 'v1', 'v2', 'v1-v2 95% 구간', '판단'], [
    ['정밀도', f3(t1['precision']), f3(t2['precision']), rng_txt(c5[0][0], c5[1][0]), verdict(c5[0][0], c5[1][0])],
    ['재현율', f3(t1['recall']), f3(t2['recall']), rng_txt(c5[0][1], c5[1][1]), verdict(c5[0][1], c5[1][1])],
    ['F1 (IoU 0.5)', f3(t1['f1']), f3(t2['f1']), rng_txt(c5[0][2], c5[1][2]), verdict(c5[0][2], c5[1][2])],
    ['F1 (IoU 0.3)', f3(I1['test']['chosen']['f1']), f3(I2['test']['chosen']['f1']), rng_txt(c3[0][2], c3[1][2]), verdict(c3[0][2], c3[1][2])],
    ['F1 (IoU 0.5, 추가 라벨 제외)', f3(round(an5[2], 4)), f3(round(bn5[2], 4)), rng_txt(cn5[0][2], cn5[1][2]), verdict(cn5[0][2], cn5[1][2])],
    ['F1 (IoU 0.3, 추가 라벨 제외)', f3(round(an3[2], 4)), f3(round(bn3[2], 4)), rng_txt(cn3[0][2], cn3[1][2]), verdict(cn3[0][2], cn3[1][2])],
    ['사진당 봉오리 개수 오차', f2(e1['count_mae_per_photo'][0]), f2(e2['count_mae_per_photo'][0]), rng_txt(cc[0][0], cc[1][0]), '**v1 우세**' if cc[1][0] < 0 else '**v2 우세**' if cc[0][0] > 0 else '확실하지 않음'],
    ['사진당 개화중 개수 오차', f2(e1['count_mae_per_photo'][1]), f2(e2['count_mae_per_photo'][1]), rng_txt(cc[0][1], cc[1][1]), '확실하지 않음'],
    ['사진당 만개 개수 오차', f2(e1['count_mae_per_photo'][2]), f2(e2['count_mae_per_photo'][2]), rng_txt(cc[0][2], cc[1][2]), '확실하지 않음'],
    ['매크로 F1 (단계까지)', f3(e1['macro_f1']), f3(e2['macro_f1']), f"v1 {f3(e1['macro_f1_photo_ci95'][0])}~{f3(e1['macro_f1_photo_ci95'][1])}, v2 {f3(e2['macro_f1_photo_ci95'][0])}~{f3(e2['macro_f1_photo_ci95'][1])}", '구간이 겹침'],
    [f"봉오리 수 (라벨 {e1['count_total_label'][0]})", e1['count_total_pred'][0], e2['count_total_pred'][0], '-', 'v1이 라벨에 가까움'],
    ['오탐 / 놓침', f"{t1['fp']} / {t1['fn']}", f"{t2['fp']} / {t2['fn']}", '-', 'v2는 오탐이 적고 v1은 놓침이 적음']], (1, 2, 3))
assert (cc[1][1] > 0 > cc[0][1]) and (cc[1][2] > 0 > cc[0][2])
T_VAL = (['항목', 'v1', 'v2'], [['정밀도', f3(v1['precision']), f3(v2['precision'])], ['재현율', f3(v1['recall']), f3(v2['recall'])], ['F1 (IoU 0.5)', f3(v1['f1']), f3(v2['f1'])], ['F1 (IoU 0.3)', f3(I1['val']['chosen']['f1']), f3(I2['val']['chosen']['f1'])], ['매크로 F1', f3(ve1['macro_f1']), f3(ve2['macro_f1'])], ['사진당 봉오리 개수 오차', f2(ve1['count_mae_per_photo'][0]), f2(ve2['count_mae_per_photo'][0])]], (1, 2))
SEG = [('긴 변 80px 미만', '<80px'), ('80~160px', '80-160px'), ('160px 이상', '>=160px'), ('봉오리', '봉오리'), ('개화중', '개화중'), ('만개', '만개'), ('빨간 꽃 구간', 'test_red'), ('노란 꽃 구간', 'test_yellow'), ('혼합 구간', 'test_mixed'), ('S25 Ultra', 'S25U'), ('S24 FE', 'S24FE')]
T_SEG = (['구분', '라벨 수', 'v1 재현율', 'v2 재현율', '차이'], [[n, t1['recall_by'][k][1], f3(t1['recall_by'][k][0]), f3(t2['recall_by'][k][0]), f"{t1['recall_by'][k][0] - t2['recall_by'][k][0]:+.3f}"] for n, k in SEG], (1, 2, 3, 4))
T_ST = (['단계', 'v1 재현율', 'v2 재현율', 'v1 정밀도', 'v2 정밀도'], [[s, f3(e1['recall'][i]), f3(e2['recall'][i]), f3(e1['precision'][i]), f3(e2['precision'][i])] for i, s in enumerate(STAGES)], (1, 2, 3, 4))
T_ARCH = (['항목', 'v1', 'v2'], [['구조', 'YOLO26s 박스 탐지 + 박스별 분류', 'YOLO26s-seg + ID + ID별 분류'], ['꽃마다 마스크', '없음', '있음'], ['ID 부여', '아직 없음 (붙일 수 있음)', '구현됨'], ['단계 분류기', '§cls/effnetv2s_r2§', '같음'], ['가중치 파일', f"{TM['v1']['size_mb']}MB", f"{TM['v2']['size_mb']}MB"], ['파라미터 (추론 기준)', f"{TM['v1']['params_fused'] / 1e4:,.0f}만", f"{TM['v2']['params_fused'] / 1e4:,.0f}만"], [f"모델 추론 (사진 1장, 화면 {TM['v1']['views']}개, V100)", f"{TM['v1']['median_s']:.3f}초", f"{TM['v2']['median_s']:.3f}초"]], ())
pm1 = {r['photo']: r for r in t1['per_photo']}; pm2 = {r['photo']: r for r in t2['per_photo']}
T_PH = (['사진', '구간', '라벨', 'v1 찾음', 'v2 찾음', 'v1 오탐', 'v2 오탐', 'v1 놓침', 'v2 놓침', '봉오리 라벨', 'v1 봉오리', 'v2 봉오리'], [[f'§{f[9:15]}§', pm1[f]['block'].replace('test_', ''), pm1[f]['gt'], pm1[f]['tp'], pm2[f]['tp'], pm1[f]['fp'], pm2[f]['fp'], pm1[f]['gt'] - pm1[f]['tp'], pm2[f]['gt'] - pm2[f]['tp'], p1[f]['label'][0], p1[f]['pred'][0], p2[f]['pred'][0]] for f in PH if pm1[f]['gt'] or pm1[f]['fp'] or pm2[f]['fp']], tuple(range(2, 12)))
JF = {(r['photo'], r['ann_id']) for r in read(V1 / 'labels/review_test_errors_r2.csv') if r['kind'] == 'fn'}; only2 = C.Counter(); both = only1 = 0
for f in PH:
    m1 = {r['ann_id'] for r in match(f, P1)[2]}; m2 = {r['ann_id'] for r in match(f, P2)[2]}; both += len(m1 & m2); only1 += len(m1 - m2)
    for r in GT[f]:
        if r['ann_id'] in m2 - m1:
            only2['added' if r['ann_id'][0] in 'TV' else 'judged' if (f, r['ann_id']) in JF else 'unjudged'] += 1
T_MISS = (['놓침 비교', '개수'], [['둘 다 놓침', both], ['v1만 놓침', only1], ['v2만 놓침', sum(only2.values())], ['그중 판정으로 추가한 봉오리', only2['added']], ['그중 칸 판정을 받지 않은 라벨', only2['unjudged']]], (1,))
COLORS = dict(tp=(40, 220, 80), fp=(255, 40, 40), fn=(60, 140, 255))


def overlay(f, P, name, width=640):
    tp, fp, fn = match(f, P); im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); z = width / im.width; im = im.resize((width, round(im.height * z)), Image.BILINEAR); d = ImageDraw.Draw(im)
    for L, c in ((tp, 'tp'), (fp, 'fp'), ([box(r) for r in fn], 'fn')):
        for b in L:
            d.rectangle([b[0] * z - 2, b[1] * z - 2, b[2] * z + 2, b[3] * z + 2], outline=COLORS[c], width=2)
    im.save(FIG / name, quality=80); return len(tp), len(fp), len(fn)


PAIRS = [('20261030_140433.jpg', 'v2가 봉오리를 더 놓친 사진'), ('20261030_135048.jpg', '노란 꽃 구간의 작은 봉오리'), ('20261030_134905.jpg', '큰 꽃은 v2가 찾고, 다른 라벨은 v1이 찾은 사진'), ('20261030_135052.jpg', '봉오리가 많은 사진')]; FIGS = []
for f, cap in PAIRS:
    s = f[9:15]; FIGS.append((s, cap, overlay(f, P1, f'cmp_{s}_v1.jpg'), overlay(f, P2, f'cmp_{s}_v2.jpg')))
SUMMARY = ['**확실한 차이는 test의 세 가지입니다.** v2는 정밀도가 높고, v1은 IoU 0.3 F1과 사진당 봉오리 개수가 낫습니다.', f"**IoU 0.5 F1은 차이가 확실하지 않습니다.** v1 {f3(t1['f1'])}, v2 {f3(t2['f1'])}이고, 구간이 0을 포함합니다.", f"**val에서는 거의 같습니다.** F1이 v1 {f3(v1['f1'])}, v2 {f3(v2['f1'])}이고, 봉오리 개수 오차도 {f2(ve1['count_mae_per_photo'][0])}, {f2(ve2['count_mae_per_photo'][0])}입니다.", f"**v2의 약점은 작은 봉오리이고 강점은 큰 꽃입니다.** 긴 변 80px 미만 재현율은 v1 {f3(t1['recall_by']['<80px'][0])}, v2 {f3(t2['recall_by']['<80px'][0])}이고, 160px 이상은 {f3(t1['recall_by']['>=160px'][0])}, {f3(t2['recall_by']['>=160px'][0])}입니다.", '**최종 모델은 아직 정하지 않았습니다.**']
HOW = ['**같은 조건:** 같은 test, val 사진과 같은 라벨로 평가했고, 임계값은 val F1로 정한 0.65로 두 모델이 같습니다. IoU 0.5 이상이면 맞은 것으로 칩니다.', '**구간:** test 17장을 복원 추출해 4,000번 계산한 v1-v2 차이의 95% 구간입니다. 0을 포함하면 우연일 수 있어 "확실하지 않음"으로 적었습니다. 사진 수가 적어 대략적인 값입니다.', '**추가 라벨 제외:** 판정으로 추가한 봉오리 44개 중 43개의 박스가 v1 예측이라, 넣으면 v1에 유리합니다. 다만 사람이 확인한 진짜 봉오리라 빼면 v2에 유리합니다. 실제는 그 사이로 봅니다.', '**값의 출처:** 표의 값은 평가 결과 파일에서 읽었고, "추가 라벨 제외"와 구간은 저장된 예측으로 다시 계산했습니다. 반올림 때문에 v1 오탐이 1개 다를 수 있습니다.']
LIM = [f"**v2만 놓친 {sum(only2.values())}개 중 {only2['unjudged']}개는 칸 판정을 받지 않았습니다.** test 사진 검수는 거쳤지만 잎이 섞여 있으면 v2의 재현율과 봉오리 개수 오차가 실제보다 나쁘게 나옵니다.", '**test 17장, val 8장 모두 휴대폰 사진입니다.** 큐브의 ESP32 카메라에서는 꽃 크기와 화질이 달라 이 비교가 바뀔 수 있습니다.', '**학습은 각 모델 한 번뿐입니다.** 씨앗을 바꾼 반복 학습이 없어 작은 차이는 우연일 수 있습니다.', '**마스크 정확도는 평가하지 못했습니다.** 사람이 그린 마스크 정답이 없어, 이 페이지의 비교는 모두 박스 기준입니다.', '**Jetson 추론 시간은 재지 않았습니다.** 표의 시간은 V100에서 잰 모델 추론만이고, 사진 열기, 화면 준비, 결과 합치기는 뺐습니다.']
T_DEC = (['방향', '장점', '단점'], [['**v2 유지, 재현율 개선** (Claude의 추천)', '요청한 세그멘테이션과 ID 구조 그대로이고 꽃마다 마스크가 있음', '재학습이 필요하고 나아진다는 보장이 없음'], ['**v1 사용, 박스에 ID 부여**', '지금 바로 봉오리를 더 잘 셈', '마스크가 없음. 단계 분류는 지금도 마스크를 쓰지 않으므로, 쓸 계획이 없다면 현실적인 선택']], ())
(OUT / 'v1_v2.json').write_text(json.dumps(dict(test=T_TEST[:2], only_v2=only2, only_v1=only1, both=both, timing=TM), ensure_ascii=False, indent=1), encoding='utf-8')


def inline(s):
    s = re.sub(r'§([^§]+)§', r'<code>\1</code>', str(s))
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)


def html_table(head, rows, num=()):
    esc = lambda c: inline(str(c).replace('<', '&lt;').replace('>=', '≥'))
    th = ''.join(f'<th class="n">{esc(h)}</th>' if i in num else f'<th>{esc(h)}</th>' for i, h in enumerate(head)); tr = ''.join('<tr>' + ''.join(f'<td class="n">{esc(c)}</td>' if i in num else f'<td>{esc(c)}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="tbl"><table><tr>{th}</tr>{tr}</table></div>'


def md_table(head, rows, num=()):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---:' if i in num else '---' for i in range(len(head))) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


uri = lambda n: 'data:image/jpeg;base64,' + base64.b64encode((FIG / n).read_bytes()).decode(); ul = lambda it: '<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in it) + '</ul>'; sec = lambda n, t, b: f'<section><h2><span class="num">{n:02d}</span>{t}</h2>{b}</section>'; kp = lambda a, b: f'<div class="kpi"><b>{a}</b><span>{b}</span></div>'
LEG = '초록 맞게 찾음, 빨강 오탐, 파랑 놓침'
fig_html = ''.join(f'<h3>{inline("§" + s + "§")}: {cap}</h3><div class="figs"><figure><img src="{uri(f"cmp_{s}_v1.jpg")}" alt="v1 {s}"><figcaption>v1: 찾음 {a[0]}, 오탐 {a[1]}, 놓침 {a[2]}</figcaption></figure><figure><img src="{uri(f"cmp_{s}_v2.jpg")}" alt="v2 {s}"><figcaption>v2: 찾음 {b[0]}, 오탐 {b[1]}, 놓침 {b[2]}</figcaption></figure></div>' for s, cap, a, b in FIGS)
STYLE = '<style>' + (V1 / 'report/style.css').read_text(encoding='utf-8') + 'td code,li code{word-break:normal;white-space:nowrap}</style>'
TITLE = '꽃 모델 v1, v2 비교'
body = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 비교</title>{STYLE}</head><body><div class="wrap"><header><div class="eyebrow">Helpai · 스마트큐브 꽃 상태 확인 · 2026-11-04</div><h1>{TITLE}</h1><p class="lead">박스 탐지(v1)와 세그멘테이션(v2)을 같은 사진과 같은 라벨로 비교했습니다. 확실한 차이는 적고, v2는 오탐이 적고 v1은 봉오리를 더 잘 셉니다.</p><div class="meta"><span>v1: {inline('§det/third_pseudo§')}</span><span>v2: {inline('§seg/first§')}</span><span>test {len(PH)}장, val {len(M1['val']['chosen']['per_photo'])}장, 휴대폰 사진</span></div><div class="nav"><a href="REPORT.html">v2 리포트</a><a href="../v1_box/REPORT.html">v1 리포트</a></div></header>
{sec(0, '요약', '<div class="kpis">' + kp(f"{f3(t1['precision'])} / {f3(t2['precision'])}", 'test 정밀도 (v1 / v2)') + kp(f"{f3(t1['recall'])} / {f3(t2['recall'])}", 'test 재현율 (v1 / v2)') + kp(f"{f3(t1['f1'])} / {f3(t2['f1'])}", 'test F1 (v1 / v2)') + kp(f"{f2(e1['count_mae_per_photo'][0])} / {f2(e2['count_mae_per_photo'][0])}", '사진당 봉오리 개수 오차 (v1 / v2)') + '</div>' + ul(SUMMARY))}
{sec(1, 'test 비교', html_table(*T_TEST) + f'<p>사진당 봉오리 개수는 v1이 {better[0]}장, v2가 {better[1]}장에서 라벨에 더 가깝고, {better[2]}장은 같습니다.</p>' + ul(HOW))}
{sec(2, 'val 비교', html_table(*T_VAL))}
{sec(3, '어디서 차이가 나는가', '<h3>구분별 탐지 재현율 (test)</h3>' + html_table(*T_SEG) + '<h3>단계까지 맞힌 값 (test)</h3>' + html_table(*T_ST) + '<h3>놓친 라벨 (test)</h3>' + html_table(*T_MISS))}
{sec(4, '구조와 속도', html_table(*T_ARCH))}
{sec(5, '사진별 비교 (test)', html_table(*T_PH))}
{sec(6, '같은 사진으로 보기', f'<p>{LEG}. 판정으로 라벨을 고친 뒤의 결과입니다.</p>' + fig_html)}
{sec(7, '한계', ul(LIM))}
{sec(8, '선택지', html_table(*T_DEC) + '<p>결정은 사용자가 합니다.</p>')}
<footer>표의 수치는 scripts/33_compare_page.py가 평가 결과 파일과 저장된 예측에서 읽어 만들었습니다.</footer></div></body></html>"""
(VER / 'COMPARE.html').write_text(body, encoding='utf-8')
md = [f'# {TITLE}', '', '> 그림을 포함한 페이지: [COMPARE.html](COMPARE.html)', '', '## 요약'] + [f'- {x}' for x in SUMMARY] + ['', '## test 비교', md_table(*T_TEST), '', f'사진당 봉오리 개수는 v1이 {better[0]}장, v2가 {better[1]}장에서 라벨에 더 가깝고, {better[2]}장은 같습니다.', ''] + [f'- {x}' for x in HOW] + ['', '## val 비교', md_table(*T_VAL), '', '## 어디서 차이가 나는가', md_table(*T_SEG), '', md_table(*T_ST), '', md_table(*T_MISS), '', '## 구조와 속도', md_table(*T_ARCH), '', '## 사진별 비교 (test)', md_table(*T_PH), '', '## 같은 사진으로 보기', LEG, ''] + [f'![{s} v1](report/cmp_{s}_v1.jpg)\n![{s} v2](report/cmp_{s}_v2.jpg)\n' for s, _, _, _ in FIGS] + ['## 한계'] + [f'- {x}' for x in LIM] + ['', '## 선택지', md_table(*T_DEC), '', '결정은 사용자가 합니다.', '']
(VER / 'COMPARE.md').write_text('\n'.join(md).replace('§', chr(96)), encoding='utf-8'); print('html', (VER / 'COMPARE.html').stat().st_size, 'leftover', body.count('§'), dict(only2), only1, both, better, FIGS)
