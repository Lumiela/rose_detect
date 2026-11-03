# v1 리포트를 만든다. 수치는 평가 결과 파일에서 읽고, 그림은 report/에 저장해 REPORT.html에 넣는다. 본문의 §는 코드 표시다.
# 사용: python 52_report.py
import base64, collections as C, json, re
from decimal import Decimal, ROUND_HALF_UP
from common import *
from PIL import Image, ImageDraw, ImageOps

R = VER / 'runs'; FIG = VER / 'report'; FIG.mkdir(exist_ok=True)
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
DET = dict(first=J(R / 'det/first/eval_r5/photo_metrics.json'), second=J(R / 'det/second/eval/photo_metrics.json'), third=J(R / 'det/third_pseudo/eval/photo_metrics.json'), fourth=J(R / 'det/fourth_pseudo/eval/photo_metrics.json'))
E2E = dict(first=J(R / 'det/first/eval_r5/e2e_metrics.json'), third=J(R / 'det/third_pseudo/eval/e2e_metrics.json'), fourth=J(R / 'det/fourth_pseudo/eval/e2e_metrics.json'))
CLS = J(R / 'cls/effnetv2s_r2/metrics.json'); man = {m['photo']: m for m in read(MANIFEST)}; ann = read(ANN); ign = read(IGN)
NAME = dict(first='§det/first§', second='§det/second§', third='§det/third_pseudo§', fourth='§det/fourth_pseudo§')
DATA = dict(first='train 13장 초안', second='train 13장 보완', third='train 108장, 임시 라벨 3,380개', fourth='train 108장, 임시 라벨 4,171개 (2회차)')
ch = lambda k, s: DET[k][s]['chosen']


def groups():
    g = {'test 17장': [], 'val 8장': [], 'train 13장 (전체 라벨)': [], 'train 95장 (일부 라벨)': []}
    print('분할 밖 라벨', C.Counter(man[r['photo']]['split'] for r in ann if man[r['photo']]['split'] not in ('train', 'val', 'test')))
    for r in ann:
        m = man[r['photo']]
        if m['split'] not in ('train', 'val', 'test'):
            continue
        k = 'test 17장' if m['split'] == 'test' else 'val 8장' if m['split'] == 'val' else 'train 95장 (일부 라벨)' if m['label_status'] == 'partial' else 'train 13장 (전체 라벨)'
        g[k].append(r['stage'])
    return [[k] + [str(v.count(s)) for s in STAGES] + [str(len(v))] for k, v in g.items()]


def overlay(key, pdir, stem, width=640):
    f = next(p for p in man if stem in p); conf = ch(key, 'val')['conf']; P = [p for p in J(R / pdir / 'preds_test.json')[f] if p[4] >= conf]
    gt = [box(r) for r in ann if r['photo'] == f]; igs = [box(r) for r in ign if r['photo'] == f]
    im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); z = width / im.width; im = im.resize((width, round(im.height * z)), Image.BILINEAR); d = ImageDraw.Draw(im); used = set(); n = C.Counter()

    def iou(a, b):
        w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

    def rect(b, c):
        d.rectangle([b[0] * z - 2, b[1] * z - 2, b[2] * z + 2, b[3] * z + 2], outline=c, width=2)

    for p in sorted(P, key=lambda q: -q[4]):
        v = [iou(p, g) if j not in used else 0 for j, g in enumerate(gt)]; j = max(range(len(v)), key=v.__getitem__) if v else -1
        if j >= 0 and v[j] >= 0.5:
            used.add(j); rect(p, (40, 220, 80)); n['tp'] += 1
        elif not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in igs):
            rect(p, (255, 40, 40)); n['fp'] += 1
    for j, g in enumerate(gt):
        if j not in used:
            rect(g, (60, 140, 255)); n['fn'] += 1
    name = f'{key}_{stem}.jpg'; im.save(FIG / name, quality=82); return name, n


figs = {k: overlay(*k) for k in [('first', 'det/first/eval_r5', '135048'), ('third', 'det/third_pseudo/eval', '135048'), ('third', 'det/third_pseudo/eval', '135052')]}
figs = {k[0] + k[2]: v for k, v in figs.items()}; print({k: dict(v[1]) for k, v in figs.items()})
sh = Image.open(R / 'det/third_pseudo/pseudo_sample/sheet_pseudo_sample.jpg'); sh.crop((0, 0, sh.width, 450)).save(FIG / 'pseudo_sample_top.jpg', quality=82)


def uri(n):
    return 'data:image/jpeg;base64,' + base64.b64encode((FIG / n).read_bytes()).decode()


def inline(s):
    s = re.sub(r'§([^§]+)§', r'<code>\1</code>', s)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)


def html_table(head, rows, num=()):
    th = ''.join(f'<th class="n">{inline(h)}</th>' if i in num else f'<th>{inline(h)}</th>' for i, h in enumerate(head))
    tr = ''.join('<tr>' + ''.join(f'<td class="n">{inline(str(c))}</td>' if i in num else f'<td>{inline(str(c))}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="tbl"><table><tr>{th}</tr>{tr}</table></div>'


def md_table(head, rows, num=()):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---:' if i in num else '---' for i in range(len(head))) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


f3 = lambda v: str(Decimal(str(v)).quantize(Decimal('0.001'), ROUND_HALF_UP)); ci = lambda v: f'{f3(v[0])}~{f3(v[1])}'
T_LABEL = (['분할', '봉오리', '개화중', '만개', '합계'], groups(), (1, 2, 3, 4))
T_DET = (['모델', '학습 데이터', '임계값', 'val F1', 'test 정밀도', 'test 재현율', 'test F1'], [[NAME[k], DATA[k], f"{ch(k, 'val')['conf']:.2f}", f3(ch(k, 'val')['f1']), f3(ch(k, 'test')['precision']), f3(ch(k, 'test')['recall']), f3(ch(k, 'test')['f1'])] for k in DET], (2, 3, 4, 5, 6))
BY = ['<80px', '80-160px', '>=160px', 'test_yellow', 'test_red', 'test_mixed', '봉오리', '개화중', '만개']
T_BY = (['구분', '라벨 수', '§first§ 재현율', '§third_pseudo§ 재현율'], [[b, ch('first', 'test')['recall_by'][b][1], f3(ch('first', 'test')['recall_by'][b][0]), f3(ch('third', 'test')['recall_by'][b][0])] for b in BY], (1, 2, 3))
c = CLS['test']; v = CLS['val']
T_CLS = (['단계', 'test 재현율', 'test 정밀도', 'test F1', 'val F1'], [[s, f3(c['recall'][i]), f3(c['precision'][i]), f3(c['f1'][i]), f3(v['f1'][i])] for i, s in enumerate(STAGES)], (1, 2, 3, 4))
T_CM = (['라벨 \\ 예측'] + STAGES, [[s] + c['confusion'][i] for i, s in enumerate(STAGES)], (1, 2, 3))
mae = lambda k: ' / '.join(f'{x:.2f}' for x in E2E[k]['test']['count_mae_per_photo'])
T_E2E = (['탐지 모델', 'val 매크로 F1', 'test 매크로 F1', 'test 95% 구간', 'test 봉오리 수 (라벨 440)', 'test 사진당 개수 오차 (봉오리 / 개화중 / 만개)'], [[NAME[k], f3(E2E[k]['val']['macro_f1']), f3(E2E[k]['test']['macro_f1']), ci(E2E[k]['test']['macro_f1_photo_ci95']), E2E[k]['test']['count_total_pred'][0], mae(k)] for k in E2E], (1, 2, 3, 4, 5))
e = E2E['third']['test']
T_E3 = (['단계', '재현율', '정밀도', 'F1'], [[s, f3(e['recall'][i]), f3(e['precision'][i]), f3(e['f1'][i])] for i, s in enumerate(STAGES)], (1, 2, 3))
cm4 = e['confusion']; T_E4 = (['라벨 \\ 예측'] + STAGES + ['없음 (놓침)'], [[n] + cm4[i] for i, n in enumerate(STAGES + ['라벨 없음 (오탐)'])], (1, 2, 3, 4))
T_REVIEW = (['대상', '판정 수', '반영'], [['test 오류 시트', '오탐 92, 미탐 141', '봉오리 42개 추가, 잎 8개 삭제, 제외 영역 24개 등'], ['val 오류 시트', '오탐 87, 미탐 29', '봉오리 28개 추가, 잎 7개 삭제, 제외 영역 18개 등'], ['train 13장 오탐 시트', '121 (전체 123)', '라벨 추가와 제외 영역'], ['임시 라벨 표본', '100', '봉오리 94개 추가, 제외 영역 5개, 차단 1개']], (1,))
LIMITS = ['**휴대폰 사진 기준입니다.** 하루, 한 농장에서 찍었고 큐브 내부 ESP32 사진으로는 평가하지 않았습니다. 타일 크기 1280도 2000×1500 사진에서 다시 확인해야 합니다.',
          '**test는 화분 단위로 분리되지 않았습니다.** 시각 구간으로 나눴고 같은 품종과 인접 재배대가 train에 있어, 점수가 실제보다 높을 수 있습니다.',
          '**§first§와 §third_pseudo§ 비교는 §first§에 유리합니다.** test와 val에 추가한 라벨 일부가 §first§의 예측 박스이고, 사람이 본 오탐도 §first§의 것입니다. §third_pseudo§의 오류 시트(오탐 86개, 미탐 85개)는 아직 판정 전입니다.',
          '**임시 라벨은 대부분 사람이 보지 않았습니다.** 3,380개 중 100개만 판정했고, 정밀도만 추정했습니다. 임시 라벨이 놓친 꽃은 알 수 없습니다.',
          '**분류기 개선은 확정할 수 없습니다.** test는 올랐지만 val은 내려갔고 구간이 겹칩니다. 개화중, 만개는 test 기준 35개, 44개뿐입니다.',
          '**추론 시간은 재지 않았습니다.** Jetson이나 큐브 환경 측정도 없습니다.',
          '**배포본이 없습니다.** 평가 스크립트만 있고, 사진을 넣으면 결과를 내는 실행 패키지는 만들지 않았습니다.']
NEXT = ['§v2_seg§: 세그멘테이션, ID 부여, ID별 분류 구조로 전환', '§third_pseudo§ test 오류 시트 판정으로 비교 편향 줄이기', '개화중 분류 개선 (이번 연결 평가에서 가장 약한 단계)', '꽃이 찍힌 ESP32 사진 확보와 독립 평가']
INFO = [['유형', '데이터 + 모델 (꽃 모델의 첫 버전)'], ['작업일', '2026-11-02 ~ 03'], ['후보 모델', '§det/third_pseudo§ + §cls/effnetv2s_r2§'], ['상태', '**후보 선정, 최종 확정 전** (test 오류 시트 판정과 ESP32 평가 남음)'], ['옛 이름', '§v11_flower§']]
S_FLOW = '사진 → 1280 타일과 전체 축척을 640으로 → YOLO26s 박스 탐지 (§flower§ 1클래스) → 원본 좌표로 합치기 → 긴 변 1.25배 정사각 크롭 → EfficientNetV2-S → 봉오리 / 개화중 / 만개'
SUM = ['농장 사진 143장 중 133장으로 꽃을 찾고 개화 단계를 판정하는 2단계 모델을 처음 만들었습니다.', f"후보 조합의 test 탐지 F1은 {f3(ch('third', 'test')['f1'])}, 단계까지 맞힌 매크로 F1은 {f3(e['macro_f1'])}입니다.", f"라벨이 일부만 있던 train 95장에 임시 라벨을 더한 결과, 재현율은 {f3(ch('first', 'test')['recall'])} → {f3(ch('third', 'test')['recall'])}, 사진당 봉오리 개수 오차는 {E2E['first']['test']['count_mae_per_photo'][0]:.2f} → {e['count_mae_per_photo'][0]:.2f}, 정밀도는 {f3(ch('first', 'test')['precision'])} → {f3(ch('third', 'test')['precision'])}입니다.", '**요청받은 세그멘테이션과 ID 구조가 아닙니다.** 박스별로 바로 분류하며, 세그멘테이션과 ID는 §v2_seg§에서 구현합니다.']
PSEUDO = ['**판정 결과:** 봉오리 94, 흐려서 알 수 없음 3, 사진 가장자리에서 잘린 봉오리 2, 잎 1', '**꽃이나 봉오리인 비율:** 96% (95% 구간 90.2~98.4%)', '**전부 봉오리인 표본:** 기존 라벨과 겹치는 예측을 뺐기 때문으로 보이며, 후보 전체를 세지는 않았습니다. 개화중, 만개 학습에 도움이 된다고 볼 수 없습니다.', '**2회차 중단:** §third_pseudo§의 예측으로 다시 만든 §fourth_pseudo§는 val과 test 모두 낮아 채택하지 않았습니다. 원인은 분석하지 않았습니다.']
F135 = next(p for p in ch('third', 'test')['per_photo'] if '135052' in p['photo'])
DET_NOTE = f"사진 §135052§ 한 장에 오탐 {F135['fp']}개가 몰려 있습니다. 이 오탐에는 라벨이 빠진 꽃이 섞여 있을 수 있어, 판정 전에는 §third_pseudo§의 정밀도를 확정할 수 없습니다."
E2E_NOTE = '매크로 F1 차이는 구간 안이라 §first§와 §third_pseudo§를 가르지 못합니다. §third_pseudo§를 후보로 고른 근거는 사진당 봉오리 개수 오차입니다. 봉오리 총수 442가 라벨 440과 비슷한 것은 오탐과 미탐이 상쇄된 결과입니다.'
CLS_NOTE = f"§cls/effnetv2s_r2§ test 정확도 {f3(c['acc'])}, 매크로 F1 {f3(c['macro_f1'])} (사진 단위 95% 구간 {ci(c['macro_f1_photo_ci95'])}). val 매크로 F1은 {f3(v['macro_f1'])}입니다. 라벨 박스로 자른 크롭 기준이라, 탐지 오류를 포함한 값은 다음 절의 연결 평가입니다."
FILES = [['§labels/annotations.csv§', '라벨 원본'], ['§labels/ignore_regions.csv§', '제외 영역'], ['§scripts/§', '내보내기, 학습, 평가, 판정 반영 스크립트'], ['§runs/det/third_pseudo/§', '후보 탐지 모델과 오류 시트 (커밋하지 않음)'], ['§runs/cls/effnetv2s_r2/§', '후보 분류 모델 (커밋하지 않음)'], ['§README.md§', '라벨 규칙, 분할, 판정 내역 상세']]

ul = lambda it: '<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in it) + '</ul>'
sec = lambda n, t, b: f'<section><h2><span class="num">{n:02d}</span>{t}</h2>{b}</section>'
flow = '<div class="flow">' + '<div class="arrow">→</div>'.join(f'<div class="step{" key" if k else ""}"><b>{a}</b><small>{b}</small></div>' for a, b, k in [('농장 사진', '휴대폰', 0), ('타일 + 전체', '1280 타일과 전체를 640으로', 0), ('YOLO26s', '박스 탐지, flower 1클래스', 1), ('합치기', '중복과 타일 조각 제거', 0), ('크롭', '긴 변 1.25배', 0), ('EfficientNetV2-S', '봉오리 / 개화중 / 만개', 1)]) + '</div>'
plain = lambda s: s.replace('§', '')
fig = lambda n, cap: f'<figure><img src="{uri(n)}" alt="{plain(cap)}"><figcaption>{inline(cap)}</figcaption></figure>'
LEG = '초록 맞게 찾음, 빨강 오탐, 파랑 놓침'
kp = lambda a, b: f'<div class="kpi"><b>{a}</b><span>{b}</span></div>'
STYLE = '<style>' + (FIG / 'style.css').read_text(encoding='utf-8') + 'td code{word-break:normal;white-space:nowrap}</style>'
TITLE = '꽃 모델 v1 리포트: 박스 탐지와 개화 단계 분류'
body = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 v1 리포트</title>{STYLE}</head><body><div class="wrap"><header><div class="eyebrow">Helpai · 스마트큐브 꽃 상태 확인 · 2026-11-02 ~ 03</div><h1>{TITLE}</h1><p class="lead">잎과 별개로 꽃을 찾고 봉오리, 개화중, 만개를 판정하는 첫 버전입니다. 비교할 이전 버전이 없어 현재 성능과 한계를 정리했습니다.</p><div class="meta"><span>후보: {inline('§det/third_pseudo§ + §cls/effnetv2s_r2§')}</span><span>평가: 휴대폰 사진, 사진 단위, IoU 0.5</span><span>최종 확정 전</span></div><div class="nav"><a href="../README.md">꽃 모델 버전 목록</a><a href="README.md">상세 기록</a></div></header>
{sec(0, '요약', '<div class="kpis">' + kp(f3(ch('third', 'test')['f1']), f"test 탐지 F1 (정밀도 {f3(ch('third', 'test')['precision'])}, 재현율 {f3(ch('third', 'test')['recall'])})") + kp(f3(e['macro_f1']), '탐지 + 단계 test 매크로 F1') + kp(f"<s>{E2E['first']['test']['count_mae_per_photo'][0]:.2f} →</s> {e['count_mae_per_photo'][0]:.2f}", '사진당 봉오리 개수 오차') + kp(f3(e['recall'][1]), '가장 약한 개화중 재현율') + '</div>' + ul(SUM) + '<div class="note">' + inline(LIMITS[0]) + '</div>')}
{sec(1, '구조', flow + '<p>' + inline('찾기와 판정을 나눈 2단계이고 잎 모델과 별개로 동작합니다. **세그멘테이션과 ID는 없습니다.** 예측은 박스 목록이고 분류는 박스마다 바로 합니다.') + '</p>')}
{sec(2, '데이터', html_table(*T_LABEL) + ul(['**분할:** 휴대폰 두 대가 같은 시각에 다른 재배대를 찍어, 시각 구간 단위로 나눴습니다. 같은 재배대로 보이는 9장은 어디에도 쓰지 않았습니다.', '**train 95장:** 2026-10-30에 꽃잎 색 후보로 판정한 박스와 표본 판정 결과만 있어 탐지 라벨로는 불완전합니다. §third_pseudo§ 학습에는 임시 라벨을 더해 썼습니다.', '**위치 정보:** S25 Ultra 사진 83장의 EXIF GPS 좌표를 지운 뒤 저장소에 넣었습니다.']) + '<h3>사람 판정</h3><p>번호를 매긴 오류 시트를 사용자가 판정해 라벨을 고쳤습니다. test 17장은 사진 전체 검수도 거쳤습니다.</p>' + html_table(*T_REVIEW))}
{sec(3, '탐지 모델 비교', html_table(*T_DET) + html_table(T_BY[0], [[str(r[0]).replace('<', '&lt;').replace('>=', '≥')] + r[1:] for r in T_BY[1]], T_BY[2]) + '<div class="figs">' + fig(figs['first135048'][0], f"§first§, 사진 §135048§: {LEG}") + fig(figs['third135048'][0], f"§third_pseudo§, 같은 사진") + '</div>' + fig(figs['third135052'][0], f"§third_pseudo§, 사진 §135052§: {LEG}") + '<div class="note">' + inline(DET_NOTE) + '</div>')}
{sec(4, '임시 라벨', fig('pseudo_sample_top.jpg', '판정에 쓴 표본 시트의 윗부분. 빨간 박스가 임시 라벨입니다.') + ul(PSEUDO))}
{sec(5, '단계 분류 (라벨 박스 기준)', '<p>' + inline(CLS_NOTE) + '</p>' + html_table(*T_CLS) + html_table(*T_CM))}
{sec(6, '탐지와 분류 연결', '<p>박스와 단계가 모두 맞아야 정답으로 칩니다.</p>' + html_table(*T_E2E) + '<h3>' + inline('§third_pseudo§ 단계별 test') + '</h3>' + html_table(*T_E3) + html_table(*T_E4) + '<div class="note">' + inline(E2E_NOTE) + '</div>')}
{sec(7, '한계', ul(LIMITS))}
{sec(8, '다음 단계', '<ol>' + ''.join(f'<li>{inline(x)}</li>' for x in NEXT) + '</ol>')}
{sec(9, '파일', html_table(['경로', '내용'], FILES))}
<footer>표의 수치는 scripts/52_report.py가 평가 결과 파일에서 읽어 만들었습니다.</footer></div></body></html>'''
(VER / 'REPORT.html').write_text(body, encoding='utf-8')
md = [f'# {TITLE}', '', '> 그림을 포함한 리포트: [REPORT.html](REPORT.html) (꽃 모델 버전 목록: [README](../README.md))', '', md_table(['항목', '내용'], INFO), '', '## 요약'] + [f'- {x}' for x in SUM] + ['', f'**{LIMITS[0].replace("**", "")}**', '', '## 구조', S_FLOW, '', '세그멘테이션과 ID는 없습니다. 예측은 박스 목록이고 분류는 박스마다 바로 합니다.', '', '## 데이터', md_table(*T_LABEL), '', md_table(*T_REVIEW), '', '## 탐지 모델 비교', md_table(*T_DET), '', md_table(*T_BY), '', '![third_pseudo, 135048](report/third_135048.jpg)', '', DET_NOTE, '', '## 임시 라벨'] + [f'- {x}' for x in PSEUDO] + ['', '## 단계 분류 (라벨 박스 기준)', CLS_NOTE, '', md_table(*T_CLS), '', '## 탐지와 분류 연결', md_table(*T_E2E), '', md_table(*T_E3), '', md_table(*T_E4), '', E2E_NOTE, '', '## 한계'] + [f'- {x}' for x in LIMITS] + ['', '## 다음 단계'] + [f'{i + 1}. {x}' for i, x in enumerate(NEXT)] + ['', '## 파일', md_table(['경로', '내용'], FILES), '']
(VER / 'REPORT.md').write_text('\n'.join(md).replace('§', chr(96)), encoding='utf-8')
print('html', (VER / 'REPORT.html').stat().st_size, 'md', (VER / 'REPORT.md').stat().st_size, 'leftover', body.count('§'))
