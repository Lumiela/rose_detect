# v1 리포트를 만든다. 수치는 평가 결과 파일에서 읽고, 그림은 report/에 저장해 REPORT.html에 넣는다. 본문의 §는 코드 표시다.
# 사용: python 52_report.py
import base64, collections as C, json, re
from decimal import Decimal, ROUND_HALF_UP
from common import *
from PIL import Image, ImageDraw, ImageOps

R = VER / 'runs'; FIG = VER / 'report'; FIG.mkdir(exist_ok=True)
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
DET = dict(first=J(R / 'det/first/eval_r7/photo_metrics.json'), second=J(R / 'det/second/eval/photo_metrics.json'), third=J(R / 'det/third_pseudo/eval/photo_metrics.json'), fourth=J(R / 'det/fourth_pseudo/eval/photo_metrics.json'))
E2E = dict(first=J(R / 'det/first/eval_r7/e2e_metrics.json'), third=J(R / 'det/third_pseudo/eval/e2e_metrics.json'), fourth=J(R / 'det/fourth_pseudo/eval/e2e_metrics.json'))
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


figs = {k: overlay(*k) for k in [('first', 'det/first/eval_r7', '135048'), ('third', 'det/third_pseudo/eval', '135048'), ('third', 'det/third_pseudo/eval', '135052')]}
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
T_E2E = (['탐지 모델', 'val 매크로 F1', 'test 매크로 F1', 'test 95% 구간', f"test 봉오리 수 (라벨 {E2E['third']['test']['count_total_label'][0]})", 'test 사진당 개수 오차 (봉오리 / 개화중 / 만개)'], [[NAME[k], f3(E2E[k]['val']['macro_f1']), f3(E2E[k]['test']['macro_f1']), ci(E2E[k]['test']['macro_f1_photo_ci95']), E2E[k]['test']['count_total_pred'][0], mae(k)] for k in E2E], (1, 2, 3, 4, 5))
e = E2E['third']['test']
T_E3 = (['단계', '재현율', '정밀도', 'F1'], [[s, f3(e['recall'][i]), f3(e['precision'][i]), f3(e['f1'][i])] for i, s in enumerate(STAGES)], (1, 2, 3))
cm4 = e['confusion']; T_E4 = (['라벨 \\ 예측'] + STAGES + ['없음 (놓침)'], [[n] + cm4[i] for i, n in enumerate(STAGES + ['라벨 없음 (오탐)'])], (1, 2, 3, 4))
T_REVIEW = (['대상', '판정 수', '반영'], [['test 오류 시트', '오탐 92, 미탐 141', '봉오리 42개 추가, 잎 8개 삭제, 제외 영역 24개 등'], ['val 오류 시트', '오탐 87, 미탐 29', '봉오리 28개 추가, 잎 7개 삭제, 제외 영역 18개 등'], ['train 13장 오탐 시트', '121 (전체 123)', '라벨 추가와 제외 영역'], ['임시 라벨 표본', '100', '봉오리 94개 추가, 제외 영역 5개, 차단 1개'], ['test 2차 오류 시트', '오탐 86, 미탐 81', '봉오리 43개 추가, 잎 3개 삭제, 제외 영역 8개 (1개는 다시 라벨로), 단계 1개 수정']], (1,))
ERR = read(R / 'det/third_pseudo/eval/errors_test.csv'); DUP = read(VER / 'labels/review_duplicates_r1.csv')
LIMITS = ['**휴대폰 사진 기준입니다.** 하루, 한 농장에서 찍었고 큐브 내부 ESP32 사진으로는 평가하지 않았습니다. 타일 크기 1280도 2000×1500 사진에서 다시 확인해야 합니다.',
          '**test는 화분 단위로 분리되지 않았습니다.** 시각 구간으로 나눴고 같은 품종과 인접 재배대가 train에 있어, 점수가 실제보다 높을 수 있습니다.',
          '**§third_pseudo§에 유리한 비교입니다.** 2차 판정에서 추가한 봉오리 43개의 박스가 §third_pseudo§의 예측이라 이 모델에서는 자동으로 정답이 되고, §first§의 점수는 내려갑니다. 추가 라벨을 뺀 비교는 §v2_seg§ 리포트에 있습니다.',
          '**임시 라벨은 대부분 사람이 보지 않았습니다.** 3,380개 중 100개만 판정했고, 정밀도만 추정했습니다. 임시 라벨이 놓친 꽃은 알 수 없습니다.',
          '**학습은 중복 라벨 정리 전 데이터로 했습니다.** 탐지 모델은 train의 중복 4쌍이 든 채로 학습했고, 분류기 단독 점수는 중복 정리와 2차 판정 전 크롭으로 잰 값입니다. 탐지 평가와 연결 평가는 최신 라벨로 다시 계산했습니다.', '**비슷한 사례가 1쌍 남아 있습니다.** train §141013§의 §D028§, §D041§은 같은 봉오리로 보이지만 IoU가 0.44라 이번 정리 대상이 아니었습니다.', '**분류기 개선은 확정할 수 없습니다.** test는 올랐지만 val은 내려갔고 구간이 겹칩니다. 개화중, 만개는 test 기준 35개, 44개뿐입니다.',
          '**추론 시간은 재지 않았습니다.** Jetson이나 큐브 환경 측정도 없습니다.',
          '**배포본이 없습니다.** 평가 스크립트만 있고, 사진을 넣으면 결과를 내는 실행 패키지는 만들지 않았습니다.']
NEXT = ['§v2_seg§: 세그멘테이션, ID 부여, ID별 분류 구조로 전환', '§v2_seg§의 오탐 18칸을 판정해 v1, v2 비교 마무리', '개화중 분류 개선 (이번 연결 평가에서 가장 약한 단계)', '꽃이 찍힌 ESP32 사진 확보와 독립 평가']
INFO = [['유형', '데이터 + 모델 (꽃 모델의 첫 버전)'], ['작업일', '2026-11-02 ~ 04'], ['후보 모델', '§det/third_pseudo§ + §cls/effnetv2s_r2§'], ['상태', '**후보 선정, 최종 확정 전** (v2 비교와 ESP32 평가 남음)'], ['옛 이름', '§v11_flower§']]
S_FLOW = '사진 → 1280 타일과 전체 축척을 640으로 → YOLO26s 박스 탐지 (§flower§ 1클래스) → 원본 좌표로 합치기 → 긴 변 1.25배 정사각 크롭 → EfficientNetV2-S → 봉오리 / 개화중 / 만개'
SUM = ['농장 사진 143장 중 133장으로 꽃을 찾고 개화 단계를 판정하는 2단계 모델을 처음 만들었습니다.', f"후보 조합의 test 탐지 F1은 {f3(ch('third', 'test')['f1'])}, 단계까지 맞힌 매크로 F1은 {f3(e['macro_f1'])}입니다.", f"라벨이 일부만 있던 train 95장에 임시 라벨을 더한 결과, 재현율은 {f3(ch('first', 'test')['recall'])} → {f3(ch('third', 'test')['recall'])}, 사진당 봉오리 개수 오차는 {E2E['first']['test']['count_mae_per_photo'][0]:.2f} → {e['count_mae_per_photo'][0]:.2f}, 정밀도는 {f3(ch('first', 'test')['precision'])} → {f3(ch('third', 'test')['precision'])}입니다.", '**요청받은 세그멘테이션과 ID 구조가 아닙니다.** 박스별로 바로 분류하며, 세그멘테이션과 ID는 §v2_seg§에서 구현합니다.']
PSEUDO = ['**판정 결과:** 봉오리 94, 흐려서 알 수 없음 3, 사진 가장자리에서 잘린 봉오리 2, 잎 1', '**꽃이나 봉오리인 비율:** 96% (95% 구간 90.2~98.4%)', '**전부 봉오리인 표본:** 기존 라벨과 겹치는 예측을 뺐기 때문으로 보이며, 후보 전체를 세지는 않았습니다. 개화중, 만개 학습에 도움이 된다고 볼 수 없습니다.', '**2회차 중단:** §third_pseudo§의 예측으로 다시 만든 §fourth_pseudo§는 val과 test 모두 낮아 채택하지 않았습니다. 원인은 분석하지 않았습니다.']
F135 = next(p for p in ch('third', 'test')['per_photo'] if '135052' in p['photo'])
DET_NOTE = f"사진 §135052§의 오탐은 판정 전 35개에서 {F135['fp']}개로 줄었습니다. 판정 전 오탐의 대부분은 라벨이 빠진 실제 봉오리였습니다."
E2E_NOTE = '매크로 F1 차이는 구간 안이라 §first§와 §third_pseudo§를 가르지 못합니다. §third_pseudo§를 후보로 고른 근거는 사진당 봉오리 개수 오차입니다. 봉오리 총수는 ' + str(E2E['third']['test']['count_total_pred'][0]) + '개로 라벨 ' + str(E2E['third']['test']['count_total_label'][0]) + '개보다 적습니다.'
CLS_NOTE = f"§cls/effnetv2s_r2§ test 정확도 {f3(c['acc'])}, 매크로 F1 {f3(c['macro_f1'])} (사진 단위 95% 구간 {ci(c['macro_f1_photo_ci95'])}). val 매크로 F1은 {f3(v['macro_f1'])}입니다. 중복 정리와 2차 판정 전 크롭으로 잰 값입니다. 라벨 박스로 자른 크롭 기준이라, 탐지 오류를 포함한 값은 다음 절의 연결 평가입니다."
FILES = [['§labels/annotations.csv§', '라벨 원본'], ['§labels/ignore_regions.csv§', '제외 영역'], ['§scripts/§', '내보내기, 학습, 평가, 판정 반영 스크립트'], ['§runs/det/third_pseudo/§', '후보 탐지 모델과 오류 시트 (커밋하지 않음)'], ['§runs/cls/effnetv2s_r2/§', '후보 분류 모델 (커밋하지 않음)'], ['§README.md§', '라벨 규칙, 분할, 판정 내역 상세']]

def iou_box(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def error_causes():
    # 놓침과 오탐을 원인별로 나눈다. 놓침: IoU 0.5 이상 예측의 신뢰도, 없으면 임계값 이상 예측과 일부 겹침(0.1~0.5) 여부
    P = J(R / 'det/third_pseudo/eval/preds_test.json'); conf = ch('third', 'val')['conf']; fn, fp = C.Counter(), C.Counter()
    for f, ps in P.items():
        G = [box(r) for r in ann if r['photo'] == f]; igs = [box(r) for r in ign if r['photo'] == f]; used = set()
        for p in sorted((q for q in ps if q[4] >= conf), key=lambda q: -q[4]):
            v = [iou_box(p, g) if j not in used else 0 for j, g in enumerate(G)]; j = max(range(len(v)), key=v.__getitem__) if v else -1
            if j >= 0 and v[j] >= 0.5:
                used.add(j)
            elif not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in igs):
                fp['near' if max([iou_box(p, g) for g in G], default=0) >= 0.1 else 'none'] += 1
        for j, g in enumerate(G):
            if j in used:
                continue
            hi = [q[4] for q in ps if iou_box(q, g) >= 0.5]
            k = ('thr' if max(hi) >= 0.3 else 'low') if hi else 'box' if any(0.1 <= iou_box(q, g) < 0.5 and q[4] >= conf for q in ps) else 'nopred'
            fn[k] += 1
    return fn, fp


FN, FP = error_causes(); print('causes', dict(FN), dict(FP))
ok = sum(cm4[i][i] for i in range(3)); wrong = sum(cm4[i][j] for i in range(3) for j in range(3) if i != j); miss = sum(cm4[i][3] for i in range(3)); N = ok + wrong + miss; assert N == sum(e['count_total_label'])
pct = lambda v: f'{v / N * 100:.1f}%'
T_RES = (['결과', '개수', '라벨 대비'], [['꽃도 찾고 단계도 맞음', ok, pct(ok)], ['꽃은 찾았지만 단계가 틀림', wrong, pct(wrong)], ['놓침', miss, pct(miss)], ['라벨에 없는 예측 (오탐)', sum(cm4[3]), '-']], (1, 2))
T_CAUSE = (['구분', '원인', '개수'], [['놓침', '신뢰도 0.3~0.65라 임계값에서 탈락', FN['thr']], ['놓침', '박스 위치가 어긋남 (IoU 0.1~0.5)', FN['box']], ['놓침', '신뢰도 0.3 미만', FN['low']], ['놓침', '예측 없음', FN['nopred']], ['오탐', '주변에 라벨 없음', FP['none']], ['오탐', '라벨과 일부만 겹침 (박스 어긋남)', FP['near']]], (2,))
sw = DET['third']['test']['sweep']
T_SWEEP = (['임계값', '정밀도', '재현율', 'F1', '오탐', '놓침'], [[k, f3(sw[k]['precision']), f3(sw[k]['recall']), f3(sw[k]['f1']), sw[k]['fp'], sw[k]['fn']] for k in ('0.5', '0.6', '0.65', '0.7', '0.8')], (1, 2, 3, 4, 5))
pm = {p['photo']: p for p in ch('third', 'test')['per_photo']}; rows_ph = []; bud = {}
for c_ in sorted(e['per_photo'], key=lambda q: q['photo']):
    p_ = pm[c_['photo']]; bud[c_['photo'][9:15]] = abs(c_['label'][0] - c_['pred'][0])
    if p_['gt'] or p_['fp']:
        rows_ph.append(['§' + c_['photo'][9:15] + '§', p_['block'], p_['gt'], p_['tp'], p_['fp'], p_['gt'] - p_['tp'], c_['label'][0], c_['pred'][0]])
T_PHOTO = (['사진', '구간', '라벨', '찾음', '오탐', '놓침', '봉오리 라벨', '봉오리 예측'], rows_ph, (2, 3, 4, 5, 6, 7))
bv = sorted(bud.values()); bud_med = bv[len(bv) // 2]; bud_wo = sum(v for k, v in bud.items() if k != '135052') / (len(bud) - 1)
print('bud err', sum(bv) / len(bv), bud_med, round(bud_wo, 2), 'rows', len(rows_ph))
for name, src, h in [('fn_top.jpg', 'sheet_test_fn.jpg', 450), ('fp_top.jpg', 'sheet_test_fp.jpg', 450)]:
    s_ = Image.open(R / 'det/third_pseudo/eval_pre_r2' / src); s_.crop((0, 0, s_.width, h)).save(FIG / name, quality=82)
rb = ch('third', 'test')['recall_by']; rb0 = ch('first', 'test')['recall_by']
EVAL_NUM = [f'**임계값:** 0.6, 0.65, 0.7의 F1이 {f3(sw["0.6"]["f1"])}, {f3(sw["0.65"]["f1"])}, {f3(sw["0.7"]["f1"])}입니다. 오탐과 놓침을 맞바꾸는 범위입니다.',
            f'**박스 어긋남:** 놓침 {FN["box"]}개, 오탐 {FP["near"]}개입니다. 2차 판정에서 오탐 23칸이 라벨이 있는 같은 봉오리로 확인됐고, 이런 칸은 놓침과 오탐에 함께 잡힙니다. IoU 기준을 0.3으로 낮추면 {J(R / "det/third_pseudo/eval_iou03/photo_metrics.json")["test"]["chosen"]["tp"] - ch("third", "test")["tp"]}개가 정답으로 바뀝니다.',
            f'**봉오리 개수 오차:** 사진당 평균 {sum(bv) / len(bv):.2f}개, 중앙값 {bud_med}개입니다. 사진 §135052§ 한 장이 {bud["135052"]}개이고, 이 사진을 빼면 평균 {bud_wo:.2f}개입니다.',
            f"**구간별 재현율:** 빨간 꽃 {f3(rb['test_red'][0])}, 혼합 {f3(rb['test_mixed'][0])}, 노란 꽃 {f3(rb['test_yellow'][0])}입니다. val에는 노란 꽃 구간이 없어 val F1({f3(ch('third', 'val')['f1'])})이 test({f3(ch('third', 'test')['f1'])})보다 높게 나온 것으로 보입니다.",
            '**단계 오류:** 대부분 인접 단계에서 납니다. 봉오리와 만개가 뒤바뀐 경우는 1개이고, 개화중 F1이 가장 낮습니다.']
EVAL_LOOK = ['**오탐 86칸:** 봉오리 66, 잎 13, 알수없음 3, 잘린 봉오리 2, 두 송이 2입니다. 봉오리 66칸 중 43칸은 라벨이 빠진 봉오리, 23칸은 라벨이 있는데 박스만 어긋난 봉오리입니다.',
             '**미탐 81칸:** 봉오리 65, 꽃 6, 잎에 가려진 꽃 3, 잎 3, 알수없음 3, 꽃 일부 1입니다. 73칸은 라벨이 맞고, 7칸은 라벨을 지우거나 제외 영역으로 옮겼습니다. §M055§는 개화중에서 봉오리로 고쳤습니다.',
             f"**큰 꽃을 놓칩니다.** §M002§, §M003§, §M008§, §M009§, §M024§는 사람이 꽃으로 확인한 놓침입니다. 긴 변 160px 이상 재현율은 §first§ {f3(rb0['>=160px'][0])}, §third_pseudo§ {f3(rb['>=160px'][0])}입니다 (라벨 {rb['>=160px'][1]}개). §M024§(사진 §135030§)는 긴 변 1,830px로 타일(1280)보다 큽니다.",
             '**붙은 꽃을 한 박스로 묶습니다.** §F002§, §F005§는 사람 판정에서도 꽃으로 나왔고, 박스 하나에 두 송이가 들어 있습니다.',
             '**Claude의 인상과 사람 판정:** §F001§~§F014§는 잎이 많다는 인상대로 잎 8칸이었고, §F015§ 이후는 봉오리가 많다는 인상대로 72칸 중 63칸이 봉오리였습니다. 하지만 잎처럼 보인다고 적은 §M020§~§M023§은 **모두 봉오리였고**, §M060§만 잎이었습니다.']
EVAL_V2 = ['큰 꽃 재현율, 붙은 꽃 분리, 사진당 개수 오차를 v2의 비교 항목으로 넣습니다.', '이번 판정은 v1의 오류만 본 것이라, v2의 오탐도 판정해야 두 모델을 공정하게 비교할 수 있습니다.', 'ESP32 사진에서 꽃이 몇 px로 찍히는지 확인해야, 작은 봉오리와 큰 꽃 중 무엇이 더 중요한지 판단할 수 있습니다.']
L_ = VER / 'labels'; ROUNDS = [('test 1차, 오탐', '§det/first§', 'review_test_fp'), ('test 1차, 미탐', '§det/first§', 'review_test_fn'), ('val, 오탐', '§det/first§', 'review_val_fp'), ('val, 미탐', '§det/first§', 'review_val_fn'), ('train 13장, 오탐', '§det/first§', 'review_train_fp'), ('임시 라벨 표본', '§det/second§ 예측', 'review_pseudo_sample_r1'), ('test 2차, 오탐', '§det/third_pseudo§', 'review_test_fp_r2'), ('test 2차, 미탐', '§det/third_pseudo§', 'review_test_fn_r2')]


def vcat(v):
    if v in ('봉오리', '잎에 가려진 봉오리'):
        return '봉오리'
    if v in ('잎', '줄기'):
        return '잎, 줄기'
    if v in ('알수없음', '제외'):
        return '알수없음'
    if v in ('꽃', '꽃 2개', '잎에 가려진 꽃') or v.startswith('개화중'):
        return '꽃'
    return '일부, 잘림'


CATS = ['봉오리', '꽃', '잎, 줄기', '알수없음', '일부, 잘림']; RJ = [(a, b, json.loads((L_ / f'{c}.json').read_text(encoding='utf-8'))) for a, b, c in ROUNDS]
T_LAB = (['판정 대상', '시트를 만든 모델'] + CATS + ['칸 수'], [[a, b] + [sum(vcat(v) == k for v in d.values()) for k in CATS] + [len(d)] for a, b, d in RJ] + [['합계', ''] + [sum(vcat(v) == k for _, _, d in RJ for v in d.values()) for k in CATS] + [sum(len(d) for _, _, d in RJ)]], (2, 3, 4, 5, 6, 7))
assert all(sum(r[2:7]) == r[7] for r in T_LAB[1]); LAB_N = T_LAB[1][-1][7]; srcn = C.Counter(r['source'] for r in ann); revn = C.Counter(r['ann_id'][0] for r in ann if r['source'] == 'review')
T_SRC = (['라벨 출처', '개수', '만든 과정'], [['§draft§', srcn['draft'], 'Claude가 격자 이미지를 보고 만든 초안 (사람 판정으로 고침)'], ['§legacy§', srcn['legacy'], '2026-10-30의 꽃잎 색 후보를 썸네일로 판정한 박스'], ['§review§ (§R§)', revn['R'], '1차 오류 시트 판정으로 추가'], ['§review§ (§P§)', revn['P'], '임시 라벨 표본 판정으로 추가'], ['§review§ (§T§)', revn['T'], '2차 오류 시트 판정으로 추가']], (1,))
LAB_STEPS = ['**1. Claude 초안:** Claude가 사진을 격자로 나눠 보고 박스와 단계를 적습니다 (§draft§).', '**2. 규칙 확인:** 작은 봉오리 포함, 개화중과 만개의 경계, 둥근 초록 대상은 봉오리, 꽃자리 없음 등은 사용자가 정했습니다.', '**3. 사진 전체 검수:** 사용자가 test 17장을 사진 단위로 봤습니다. 수정 없이 통과해 §verified§가 됐습니다.', '**4. 오류 시트 제작:** 모델이 틀린 곳을 스크립트가 번호 칸으로 잘라 줍니다. Claude는 시트를 먼저 보고 인상을 적지만, **이 인상으로 라벨을 바꾸지 않습니다.**', '**5. 사람 판정:** 사용자가 칸마다 봉오리, 꽃, 잎, 알수없음 등을 판정합니다. 라벨 수정의 근거는 이 판정입니다.', '**6. 반영과 재평가:** Claude가 칸 수를 시트와 맞춘 뒤 스크립트로 라벨에 반영하고, 평가를 다시 돌립니다. 판정 원본은 §labels/review_*.json§과 §review_*_errors_*.csv§에 남습니다.']
LAB_FIND = [f'**사용자가 판정한 칸은 모두 {LAB_N}개입니다.** test 17장 사진 검수와 라벨 규칙 결정은 이 수에 들어 있지 않습니다.', '**칸 판정이 사진 검수보다 많이 찾았습니다.** test 17장은 사진 검수에서 수정이 없었지만, 칸 판정으로 빠진 봉오리 85개(1차 42, 2차 43)를 추가하고 잎 11개(1차 8, 2차 3)를 지웠습니다.', '**Claude의 판단도 사람이 고쳤습니다.** 2차에서 Claude가 잎처럼 보인다고 적은 §M020§~§M023§은 모두 봉오리였습니다.', '**판정에서 걸러지지 않은 오류도 있었습니다.** Claude 초안의 중복 박스 10쌍은 사진 검수를 통과했고, ID 결과를 라벨과 맞춰 보다가 뒤늦게 찾았습니다. 사용자 승인 후 지웠습니다.', '**판정한 번호는 그 회차 시트의 번호입니다.** 회차마다 번호가 새로 매겨지므로 다른 회차의 같은 번호와는 다른 칸입니다.']
LAB_NOT = ['**train 95장:** 사람이 사진 단위로 보지 않았고, 임시 라벨 3,380개 중 100개만 판정했습니다.', '**val 8장, train 13장:** 사진 단위 검수 없이 오류 시트로만 고쳤습니다. train 13장의 미탐 시트는 판정하지 않았습니다.', '**§v2_seg§의 마스크:** SAM 초안이고 사람 판정이 없습니다.', '**§v2_seg§의 오탐:** 18칸이 아직 판정 전입니다.']



ul = lambda it: '<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in it) + '</ul>'
sec = lambda n, t, b: f'<section><h2><span class="num">{n:02d}</span>{t}</h2>{b}</section>'
flow = '<div class="flow">' + '<div class="arrow">→</div>'.join(f'<div class="step{" key" if k else ""}"><b>{a}</b><small>{b}</small></div>' for a, b, k in [('농장 사진', '휴대폰', 0), ('타일 + 전체', '1280 타일과 전체를 640으로', 0), ('YOLO26s', '박스 탐지, flower 1클래스', 1), ('합치기', '중복과 타일 조각 제거', 0), ('크롭', '긴 변 1.25배', 0), ('EfficientNetV2-S', '봉오리 / 개화중 / 만개', 1)]) + '</div>'
plain = lambda s: s.replace('§', '')
fig = lambda n, cap: f'<figure><img src="{uri(n)}" alt="{plain(cap)}"><figcaption>{inline(cap)}</figcaption></figure>'
LEG = '초록 맞게 찾음, 빨강 오탐, 파랑 놓침'
kp = lambda a, b: f'<div class="kpi"><b>{a}</b><span>{b}</span></div>'
STYLE = '<style>' + (FIG / 'style.css').read_text(encoding='utf-8') + 'td code,li code{word-break:normal;white-space:nowrap}details{margin:6px 0}summary{cursor:pointer;font-weight:600}details p{margin:6px 0;font-size:.9em;overflow-wrap:anywhere}</style>'
TITLE = '꽃 모델 v1 리포트: 박스 탐지와 개화 단계 분류'
LAB_FLOW = '<div class="flow">' + '<div class="arrow">→</div>'.join(f'<div class="step{" key" if k else ""}"><b>{a}</b><small>{b}</small></div>' for a, b, k in [('Claude 초안', '박스와 단계 작성', 0), ('사진 검수', '사용자, test 17장', 1), ('학습과 평가', '틀린 곳 찾기', 0), ('오류 시트', '번호 칸, Claude가 먼저 확인', 0), ('사람 판정', '사용자, 칸마다', 1), ('반영', '칸 수 대조 후 라벨 수정', 0), ('재평가', '다음 회차로', 0)]) + '</div>'
LAB_DET = ''.join('<details><summary>' + inline(a) + f' ({len(d)}칸)</summary>' + ''.join('<p><b>' + v + f' ({n}):</b> ' + ', '.join(k for k, x in d.items() if x == v) + '</p>' for v, n in C.Counter(d.values()).most_common()) + '</details>' for a, _, d in RJ)
LAB_HTML = LAB_FLOW + ul(LAB_STEPS) + '<h3>사용자가 판정한 내역</h3>' + html_table(*T_LAB) + ul(LAB_FIND) + '<h3>칸별 판정</h3>' + LAB_DET + '<h3>지금 라벨의 출처</h3>' + html_table(*T_SRC) + '<h3>사람이 보지 않은 부분</h3>' + ul(LAB_NOT)
LAB_MD = ['## 라벨링: Claude 검수 후 사람 판정'] + [f'- {x}' for x in LAB_STEPS] + ['', '### 사용자가 판정한 내역', md_table(*T_LAB), ''] + [f'- {x}' for x in LAB_FIND] + ['', '### 칸별 판정'] + [f'- {a}, {v} ({n}): ' + ', '.join(k for k, x in d.items() if x == v) for a, _, d in RJ for v, n in C.Counter(d.values()).most_common()] + ['', '### 지금 라벨의 출처', md_table(*T_SRC), '', '### 사람이 보지 않은 부분'] + [f'- {x}' for x in LAB_NOT] + ['']
body = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 v1 리포트</title>{STYLE}</head><body><div class="wrap"><header><div class="eyebrow">Helpai · 스마트큐브 꽃 상태 확인 · 2026-11-02 ~ 04</div><h1>{TITLE}</h1><p class="lead">잎과 별개로 꽃을 찾고 봉오리, 개화중, 만개를 판정하는 첫 버전입니다. 비교할 이전 버전이 없어 현재 성능과 한계를 정리했습니다.</p><div class="meta"><span>후보: {inline('§det/third_pseudo§ + §cls/effnetv2s_r2§')}</span><span>평가: 휴대폰 사진, 사진 단위, IoU 0.5</span><span>최종 확정 전</span></div><div class="nav"><a href="../README.md">꽃 모델 버전 목록</a><a href="README.md">상세 기록</a></div></header>
{sec(0, '요약', '<div class="kpis">' + kp(f3(ch('third', 'test')['f1']), f"test 탐지 F1 (정밀도 {f3(ch('third', 'test')['precision'])}, 재현율 {f3(ch('third', 'test')['recall'])})") + kp(f3(e['macro_f1']), '탐지 + 단계 test 매크로 F1') + kp(f"<s>{E2E['first']['test']['count_mae_per_photo'][0]:.2f} →</s> {e['count_mae_per_photo'][0]:.2f}", '사진당 봉오리 개수 오차') + kp(f3(e['recall'][1]), '가장 약한 개화중 재현율') + '</div>' + ul(SUM) + '<div class="note">' + inline(LIMITS[0]) + '</div>')}
{sec(1, '구조', flow + '<p>' + inline('찾기와 판정을 나눈 2단계이고 잎 모델과 별개로 동작합니다. **세그멘테이션과 ID는 없습니다.** 예측은 박스 목록이고 분류는 박스마다 바로 합니다.') + '</p>')}
{sec(2, '데이터', html_table(*T_LABEL) + ul(['**분할:** 휴대폰 두 대가 같은 시각에 다른 재배대를 찍어, 시각 구간 단위로 나눴습니다. 같은 재배대로 보이는 9장은 어디에도 쓰지 않았습니다.', '**train 95장:** 2026-10-30에 꽃잎 색 후보로 판정한 박스와 표본 판정 결과만 있어 탐지 라벨로는 불완전합니다. §third_pseudo§ 학습에는 임시 라벨을 더해 썼습니다.', '**위치 정보:** S25 Ultra 사진 83장의 EXIF GPS 좌표를 지운 뒤 저장소에 넣었습니다.', f'**중복 라벨 정리 (2026-11-04):** 한 봉오리에 박스가 2개 있던 {len(DUP)}쌍에서 1개씩 지웠습니다. 모두 Claude의 초안에서 생겼고, 내역은 §labels/review_duplicates_r1.csv§에 있습니다. 예측은 그대로이고 test 놓침이 4개, val 놓침이 1개 줄었습니다.']) + '<h3>사람 판정</h3><p>' + inline('자세한 과정과 판정 내역은 다음 절에 있습니다.') + '</p>' + html_table(*T_REVIEW))}\n{sec(3, '라벨링: Claude 검수 후 사람 판정', LAB_HTML)}
{sec(4, '탐지 모델 비교', html_table(*T_DET) + html_table(T_BY[0], [[str(r[0]).replace('<', '&lt;').replace('>=', '≥')] + r[1:] for r in T_BY[1]], T_BY[2]) + '<div class="figs">' + fig(figs['first135048'][0], f"§first§, 사진 §135048§: {LEG}") + fig(figs['third135048'][0], f"§third_pseudo§, 같은 사진") + '</div>' + fig(figs['third135052'][0], f"§third_pseudo§, 사진 §135052§: {LEG}") + '<div class="note">' + inline(DET_NOTE) + '</div>')}
{sec(5, '임시 라벨', fig('pseudo_sample_top.jpg', '판정에 쓴 표본 시트의 윗부분. 빨간 박스가 임시 라벨입니다.') + ul(PSEUDO))}
{sec(6, '단계 분류 (라벨 박스 기준)', '<p>' + inline(CLS_NOTE) + '</p>' + html_table(*T_CLS) + html_table(*T_CM))}
{sec(7, '탐지와 분류 연결', '<p>박스와 단계가 모두 맞아야 정답으로 칩니다.</p>' + html_table(*T_E2E) + '<h3>' + inline('§third_pseudo§ 단계별 test') + '</h3>' + html_table(*T_E3) + html_table(*T_E4) + '<div class="note">' + inline(E2E_NOTE) + '</div>')}
{sec(8, '예측 결과 평가', '<p>' + inline('후보 조합(§third_pseudo§ + §effnetv2s_r2§)의 test 사진 17장 결과입니다.') + '</p>' + html_table(*T_RES) + html_table(*T_CAUSE) + html_table(*T_SWEEP) + ul(EVAL_NUM) + '<h3>사진별</h3>' + html_table(*T_PHOTO) + '<h3>사람 판정 결과</h3><p>' + inline('번호는 판정에 쓴 시트 기준입니다 (§labels/review_test_errors_r2.csv§).') + '</p>' + ul(EVAL_LOOK) + '<div class="figs">' + fig('fn_top.jpg', '판정에 쓴 놓침 시트 윗부분 (§M001§~§M030§)') + fig('fp_top.jpg', '판정에 쓴 오탐 시트 윗부분 (§F001§~§F030§)') + '</div><h3>v2 전에 볼 점</h3>' + ul(EVAL_V2))}
{sec(9, '한계', ul(LIMITS))}
{sec(10, '다음 단계', '<ol>' + ''.join(f'<li>{inline(x)}</li>' for x in NEXT) + '</ol>')}
{sec(11, '파일', html_table(['경로', '내용'], FILES))}
<footer>표의 수치는 scripts/52_report.py가 평가 결과 파일에서 읽어 만들었습니다.</footer></div></body></html>'''
(VER / 'REPORT.html').write_text(body, encoding='utf-8')
md = [f'# {TITLE}', '', '> 그림을 포함한 리포트: [REPORT.html](REPORT.html) (꽃 모델 버전 목록: [README](../README.md))', '', md_table(['항목', '내용'], INFO), '', '## 요약'] + [f'- {x}' for x in SUM] + ['', f'**{LIMITS[0].replace("**", "")}**', '', '## 구조', S_FLOW, '', '세그멘테이션과 ID는 없습니다. 예측은 박스 목록이고 분류는 박스마다 바로 합니다.', '', '## 데이터', md_table(*T_LABEL), '', md_table(*T_REVIEW), ''] + LAB_MD + ['## 탐지 모델 비교', md_table(*T_DET), '', md_table(*T_BY), '', '![third_pseudo, 135048](report/third_135048.jpg)', '', DET_NOTE, '', '## 임시 라벨'] + [f'- {x}' for x in PSEUDO] + ['', '## 단계 분류 (라벨 박스 기준)', CLS_NOTE, '', md_table(*T_CLS), '', '## 탐지와 분류 연결', md_table(*T_E2E), '', md_table(*T_E3), '', md_table(*T_E4), '', E2E_NOTE, '', '## 예측 결과 평가', md_table(*T_RES), '', md_table(*T_CAUSE), '', md_table(*T_SWEEP), ''] + [f'- {x}' for x in EVAL_NUM] + ['', md_table(*T_PHOTO), '', '### 사람 판정 결과', '번호는 판정에 쓴 시트 기준입니다.', ''] + [f'- {x}' for x in EVAL_LOOK] + ['', '### v2 전에 볼 점'] + [f'- {x}' for x in EVAL_V2] + ['', '## 한계'] + [f'- {x}' for x in LIMITS] + ['', '## 다음 단계'] + [f'{i + 1}. {x}' for i, x in enumerate(NEXT)] + ['', '## 파일', md_table(['경로', '내용'], FILES), '']
(VER / 'REPORT.md').write_text('\n'.join(md).replace('§', chr(96)), encoding='utf-8')
print('html', (VER / 'REPORT.html').stat().st_size, 'md', (VER / 'REPORT.md').stat().st_size, 'leftover', body.count('§'))
