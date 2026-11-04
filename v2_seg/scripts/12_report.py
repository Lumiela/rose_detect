# v2 리포트를 만든다. 수치는 평가 결과 파일과 마스크 파일에서 읽고, 그림은 report/에 저장해 REPORT.html에 넣는다. 본문의 §는 코드 표시다.
# 사용: python 12_report.py
import base64, collections as C, csv, re, shutil, subprocess
from decimal import Decimal, ROUND_HALF_UP
from common import *
from PIL import Image, ImageDraw, ImageFont, ImageOps

R = VER / 'runs' / 'mask_check'; FIG = VER / 'report'; FIG.mkdir(exist_ok=True)
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
INITIAL = 'ecb6b63'  # 재생성 전 마스크가 있는 커밋
rows = [json.loads(t) for t in subprocess.run(['git', 'show', f'{INITIAL}:v2_seg/labels/masks.jsonl'], cwd=FLOWER, capture_output=True).stdout.decode('utf-8').splitlines()]; now = read_masks(); assert len(rows) - len(now) == 18  # 중복 정리 9개, 2차 판정 7개, 4차 판정 2개를 뺐다
name = lambda r: r['stage'] or '임시 라벨'; N = len(rows)
SEV = ['IoU 0.7 이상', '0.5~0.7', '0.3~0.5', '0.3 미만', '마스크 없음']
RETRIED = '622fb9d'  # 재생성은 끝났고 2차 판정으로 라벨을 지우기 전 마스크가 있는 커밋
at = [json.loads(t) for t in subprocess.run(['git', 'show', f'{RETRIED}:v2_seg/labels/masks.jsonl'], cwd=FLOWER, capture_output=True).stdout.decode('utf-8').splitlines()]
fixed = sum(1 for r in at if r.get('retry')); failed = sum(1 for r in at if r.get('use') is False)


def sev(r):
    if not r['poly']:
        return SEV[4]
    v = r['box_iou']
    return SEV[3] if v < 0.3 else SEV[2] if v < 0.5 else SEV[1] if v < 0.7 else SEV[0]


def size(r):
    s = max(r['box'][2] - r['box'][0], r['box'][3] - r['box'][1])
    return '40px 미만' if s < 40 else '40~80px' if s < 80 else '80~160px' if s < 160 else '160px 이상'


def table_by(key, order):
    d = C.defaultdict(C.Counter)
    for r in rows:
        d[key(r)][sev(r)] += 1
    out = []
    for k in order:
        n = sum(d[k].values()); bad = d[k][SEV[2]] + d[k][SEV[3]] + d[k][SEV[4]]
        out.append([k, n] + [d[k][s] for s in SEV[:2]] + [bad, f'{bad / n * 100:.1f}%'])
    return (['구분', '개수'] + SEV[:2] + ['0.5 미만 + 없음', '실패 비율'], out, (1, 2, 3, 4, 5))


cnt = C.Counter(sev(r) for r in rows); fail = cnt[SEV[2]] + cnt[SEV[3]] + cnt[SEV[4]]; assert sum(cnt.values()) == N and fixed + failed == fail
T_ALL = (['마스크 박스와 프롬프트 박스의 IoU', '개수', '비율'], [[s, cnt[s], f'{cnt[s] / N * 100:.1f}%'] for s in SEV], (1, 2))
T_STAGE = table_by(name, STAGES + ['임시 라벨']); T_SIZE = table_by(size, ['40px 미만', '40~80px', '80~160px', '160px 이상']); T_SPLIT = table_by(lambda r: r['split'], ['train', 'val', 'test']); T_SRC = table_by(lambda r: r['source'], ['draft', 'legacy', 'review', 'pseudo'])
for n_, src, h in [('sample_top.jpg', 'sample', 750), ('flagged_top.jpg', 'flagged', 750)]:
    s_ = Image.open(R / src / 'sheet_masks.jpg'); s_.crop((0, 0, s_.width, h)).save(FIG / n_, quality=84)

V1R = V1 / 'runs/det/third_pseudo'; V2R = VER / 'runs/seg/first'; D1, D2 = J(V1R / 'eval/photo_metrics.json'), J(V2R / 'eval/photo_metrics.json'); I1, I2 = J(V1R / 'eval_iou03/photo_metrics.json'), J(V2R / 'eval_iou03/photo_metrics.json'); MB = J(V2R / 'eval_maskbox/photo_metrics.json')
E1, E2 = J(V1R / 'eval/e2e_metrics.json'), J(V2R / 'eval/e2e_metrics.json'); CP = J(V2R / 'compare/compare.json')
for p in ('134905', '135052', '140440'):
    shutil.copy(V2R / 'compare' / f'v2_{p}.jpg', FIG / f'v2_{p}.jpg')
f3 = lambda v: str(Decimal(str(v)).quantize(Decimal('0.001'), ROUND_HALF_UP)); ci = lambda v: f'{f3(v[0])}~{f3(v[1])}'
c1, c2 = D1['test']['chosen'], D2['test']['chosen']
with open(V2R / 'results.csv', encoding='utf-8', newline='') as f_:
    res = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f_)]
best = max(res, key=lambda r: float(r['metrics/mAP50-95(M)']))
T_DET = (['항목', 'v1 §third_pseudo§', 'v2 §seg/first§'], [['임계값 (val F1 기준)', f"{D1['val']['chosen']['conf']:.2f}", f"{D2['val']['chosen']['conf']:.2f}"], ['val F1', f3(D1['val']['chosen']['f1']), f3(D2['val']['chosen']['f1'])], ['test 정밀도', f3(c1['precision']), f3(c2['precision'])], ['test 재현율', f3(c1['recall']), f3(c2['recall'])], ['test F1', f3(c1['f1']), f3(c2['f1'])], ['test 오탐', c1['fp'], c2['fp']], ['test 놓침', c1['fn'], c2['fn']], ['test F1 (IoU 0.3)', f3(I1['test']['chosen']['f1']), f3(I2['test']['chosen']['f1'])]], (1, 2))
BY = ['<80px', '80-160px', '>=160px', '봉오리', '개화중', '만개', 'test_red', 'test_yellow', 'test_mixed', 'S25U', 'S24FE']
T_BY = (['구분', '라벨 수', 'v1 재현율', 'v2 재현율'], [[b, c1['recall_by'][b][1], f3(c1['recall_by'][b][0]), f3(c2['recall_by'][b][0])] for b in BY], (1, 2, 3))
e1, e2 = E1['test'], E2['test']; v1e, v2e = E1['val'], E2['val']; mae = lambda e: ' / '.join(f'{x:.2f}' for x in e['count_mae_per_photo'])
T_E2E = (['항목', 'v1', 'v2'], [['test 매크로 F1', f3(e1['macro_f1']), f3(e2['macro_f1'])], ['test 95% 구간', ci(e1['macro_f1_photo_ci95']), ci(e2['macro_f1_photo_ci95'])], ['val 매크로 F1', f3(v1e['macro_f1']), f3(v2e['macro_f1'])], [f"test 봉오리 수 (라벨 {e2['count_total_label'][0]})", e1['count_total_pred'][0], e2['count_total_pred'][0]], ['test 사진당 개수 오차 (봉오리 / 개화중 / 만개)', mae(e1), mae(e2)]], (1, 2))
T_E3 = (['단계', 'v1 재현율', 'v2 재현율', 'v1 정밀도', 'v2 정밀도'], [[s, f3(e1['recall'][i]), f3(e2['recall'][i]), f3(e1['precision'][i]), f3(e2['precision'][i])] for i, s in enumerate(STAGES)], (1, 2, 3, 4))
T_E4 = (['라벨 \\ 예측'] + STAGES + ['없음 (놓침)'], [[n_] + e2['confusion'][i] for i, n_ in enumerate(STAGES + ['라벨 없음 (오탐)'])], (1, 2, 3, 4))
T_PHOTO = (['사진', '라벨', 'v1 찾음', 'v2 찾음', 'v1 오탐', 'v2 오탐', 'v1 놓침', 'v2 놓침'], [['§' + r[0] + '§'] + r[1:] for r in CP['rows']], (1, 2, 3, 4, 5, 6, 7))
pr = CP['pair']; sw = D2['test']['sweep']['0.6']
STEP = [('eval_pre_dedup', 'eval_iou03_pre_dedup'), ('eval_pre_r2', 'eval_iou03_pre_r2'), ('eval', 'eval_iou03')]; HIST = [(J(V / a / 'photo_metrics.json')['test']['chosen'], J(V / b / 'photo_metrics.json')['test']['chosen'], J(V / a / 'e2e_metrics.json')['test']) for V in (V1R, V2R) for a, b in STEP]
T_DUP = (['test 항목', 'v1 처음', 'v1 중복 정리', 'v1 판정 반영', 'v2 처음', 'v2 중복 정리', 'v2 판정 반영'], [['라벨 수'] + [a['tp'] + a['fn'] for a, _, _ in HIST], ['오탐'] + [a['fp'] for a, _, _ in HIST], ['놓침'] + [a['fn'] for a, _, _ in HIST], ['정밀도'] + [f3(a['precision']) for a, _, _ in HIST], ['재현율'] + [f3(a['recall']) for a, _, _ in HIST], ['F1 (IoU 0.5)'] + [f3(a['f1']) for a, _, _ in HIST], ['F1 (IoU 0.3)'] + [f3(b['f1']) for _, b, _ in HIST], ['매크로 F1'] + [f3(c['macro_f1']) for _, _, c in HIST], ['사진당 봉오리 개수 오차'] + [f"{c['count_mae_per_photo'][0]:.2f}" for _, _, c in HIST]], (1, 2, 3, 4, 5, 6))
DUP_NOTE = '**중복 정리:** 한 봉오리에 박스가 2개 있던 10쌍에서 1개씩 지웠습니다. 모두 Claude의 초안에서 생겼습니다. **판정 반영:** v1 §third_pseudo§의 오류 시트(오탐 86칸, 미탐 81칸), v2의 오탐 시트(18칸), v2만 놓친 라벨(35칸)을 사용자가 판정한 결과입니다. 두 경우 모두 모델의 예측과 임계값은 그대로이고 라벨만 바뀌었습니다.'
GA, GI = C.defaultdict(list), C.defaultdict(list)
for r_ in read(ANN):
    GA[r_['photo']].append(r_)
for r_ in read(IGN):
    GI[r_['photo']].append(box(r_))


def fair(P, neutral, iou=0.5, conf=0.65):
    # 사진 단위 맞춤. neutral이면 판정에서 추가한 라벨(T, V)을 제외 영역처럼 다뤄 어느 모델의 점수에도 넣지 않는다
    tp = fp = fn = 0
    for f, ps in P.items():
        G = [box(r) for r in GA[f] if not (neutral and r['ann_id'][0] in 'TV')]; I_ = GI[f] + ([box(r) for r in GA[f] if r['ann_id'][0] in 'TV'] if neutral else []); used = set()
        for p in sorted((q for q in ps if q[4] >= conf), key=lambda q: -q[4]):
            v = [ov(p[:4], g)[0] if k not in used else 0 for k, g in enumerate(G)]; k = max(range(len(v)), key=v.__getitem__) if v else -1
            if k >= 0 and v[k] >= iou:
                used.add(k); tp += 1
            elif not any(b[0] <= (p[0] + p[2]) / 2 <= b[2] and b[1] <= (p[1] + p[3]) / 2 <= b[3] for b in I_):
                fp += 1
        fn += len(G) - len(used)
    pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return [f3(round(pre, 4)), f3(round(rec, 4)), f3(round(2 * pre * rec / max(pre + rec, 1e-9), 4))]


PT1, PT2 = J(V1R / 'eval/preds_test.json'), J(V2R / 'eval/preds_test.json')
T_FAIR = (['기준', 'v1 정밀도', 'v1 재현율', 'v1 F1', 'v2 정밀도', 'v2 재현율', 'v2 F1'], [[n_] + fair(PT1, t_, u_) + fair(PT2, t_, u_) for n_, t_, u_ in [('모든 라벨, IoU 0.5', False, 0.5), ('추가 라벨 제외, IoU 0.5', True, 0.5), ('모든 라벨, IoU 0.3', False, 0.3), ('추가 라벨 제외, IoU 0.3', True, 0.3)]], (1, 2, 3, 4, 5, 6))
ADDED = [(f, box(r)) for f, v in GA.items() for r in v if r['ann_id'][0] == 'T']; hit = lambda P, c: sum(1 for f, b in ADDED if max([ov(p[:4], b)[0] for p in P.get(f, []) if p[4] >= c] or [0]) >= 0.5)
V2_ERR = read(V2R / 'eval/errors_test.csv'); V2_FP = sum(1 for r in V2_ERR if r['kind'] == 'fp')
NV = sum(1 for v in GA.values() for r in v if r['ann_id'][0] == 'V'); SW1, SW2 = D1['test']['sweep']['0.65'], D2['test']['sweep']['0.55']
FAIR_NOTE = [f'**두 모델의 오류를 모두 판정했습니다.** 빠진 봉오리가 v1 시트에서 {len(ADDED)}개, v2 시트에서 {NV}개 나왔습니다. 라벨이 빠져 v2가 손해를 본다는 가능성은 거의 사라졌습니다. v2만 놓친 라벨 35칸도 판정했는데, 32칸은 봉오리, 1칸은 개화중이었고 잎은 없었습니다.', '**그래도 v1에 조금 유리합니다.** 추가 라벨의 박스가 대부분 v1의 예측이기 때문입니다. 추가 라벨을 빼면 IoU 0.5에서는 v2가 조금 높지만, 빼는 라벨은 사람이 확인한 진짜 봉오리라 이 값은 반대로 v2에 유리합니다. 실제는 두 값 사이로 봅니다. IoU 0.3에서는 어느 쪽이든 v1이 높습니다.', f'**v2가 놓친 원인:** 추가한 봉오리 {len(ADDED)}개 중 임계값 0.65에서 {hit(PT2, 0.65)}개를 찾고, 0.3으로 낮추면 {hit(PT2, 0.3)}개를 찾습니다. 낮은 신뢰도가 원인인 경우가 많습니다.', f"**임계값을 낮춰도 v1을 넘지 못합니다.** v2의 0.55는 정밀도 {f3(SW2['precision'])}, 재현율 {f3(SW2['recall'])}이고, v1의 0.65는 {f3(SW1['precision'])}, {f3(SW1['recall'])}입니다. test를 본 참고값이라 선택에는 쓰지 않습니다.", '이 표의 값은 저장된 예측으로 다시 계산해, 위 표와 오탐이 1개 다를 수 있습니다.']
DEC = ['**모델은 아직 정하지 않았습니다.** 사용자가 결정합니다.', '**v2 유지, 재현율 개선 (Claude의 추천):** 요청한 세그멘테이션과 ID 구조 그대로이고 꽃마다 마스크가 있습니다. 재학습이 필요하고 나아진다는 보장은 없습니다.', '**v1 사용, 박스에 ID 부여:** 지금 바로 봉오리를 더 잘 세지만 마스크가 없습니다. 단계 분류는 지금도 마스크를 쓰지 않으므로, 마스크의 쓰임이 없다면 현실적인 선택입니다.', '**ESP32 사진:** 큐브 카메라에서는 꽃 크기가 달라 이 비교가 바뀔 수 있습니다.']
IDR = VER / 'runs/ids/first'; ESPR = VER / 'runs/ids/esp_cam_20261103'; CK = J(IDR / 'check.json'); FL = {p['photo']: p['flowers'] for p in J(IDR / 'flowers.json')}; SM = read(IDR / 'summary.csv'); ESP = read(ESPR / 'summary.csv'); ESPF = read(ESPR / 'flowers.csv')
GT = C.defaultdict(list)
for r_ in read(ANN):
    GT[r_['photo']].append(r_)


def crop_sheet(items, out, cols=5, cell=300):
    # 항목: (사진, ID 목록, 설명, 라벨 박스 목록). 하늘색은 예측 마스크, 파랑은 라벨 박스
    S = Image.new('RGB', (cols * cell, ((len(items) + cols - 1) // cols) * (cell + 26)), (30, 30, 30)); d = ImageDraw.Draw(S); F = ImageFont.load_default(size=17)
    for k, (f, xs, cap, gs) in enumerate(items):
        im = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert('RGB'); B = [[x['x0'], x['y0'], x['x1'], x['y1']] for x in xs] + gs
        x0, y0, x1, y1 = min(b[0] for b in B), min(b[1] for b in B), max(b[2] for b in B), max(b[3] for b in B); l = max(x1 - x0, y1 - y0) * 1.3; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; o = (int(cx - l / 2), int(cy - l / 2)); z = cell / l
        t = im.crop((o[0], o[1], int(o[0] + l), int(o[1] + l))).resize((cell, cell), Image.BILINEAR); e = ImageDraw.Draw(t)
        for g in gs:
            e.rectangle([(g[0] - o[0]) * z, (g[1] - o[1]) * z, (g[2] - o[0]) * z, (g[3] - o[1]) * z], outline=(80, 160, 255), width=2)
        for x in xs:
            q = [((a - o[0]) * z, (c - o[1]) * z) for a, c in x['poly']]; e.line(q + q[:1], fill=(0, 255, 255), width=2)
        S.paste(t, ((k % cols) * cell, (k // cols) * (cell + 26) + 26)); d.text(((k % cols) * cell + 5, (k // cols) * (cell + 26) + 4), cap, font=F, fill=(255, 255, 255))
    S.save(FIG / out, quality=84)


fid = lambda f, i: next(x for x in FL[f] if x['idx'] == i)
MERGED = [m for s_ in ('val', 'test') for m in CK[s_]['merged']]; SPLIT = [m for s_ in ('val', 'test') for m in CK[s_]['split']]
crop_sheet([(f, [fid(f, i)], f'{f[9:15]} #{i} merged', [box(GT[f][j]) for j in js]) for f, i, js in MERGED] + [(f, [fid(f, i) for i in ids], f"{f[9:15]} #{','.join(map(str, ids))} split", [box(GT[f][j])]) for f, j, ids in SPLIT], 'id_merged_split.jpg')
crop_sheet([(f, [x], f"{f[9:15]} #{x['idx']} sol {x['solidity']}", []) for f, v in FL.items() for x in v if x['solidity'] < 0.8], 'id_low_solidity.jpg')
Image.open(IDR / '20261030_134905_ids.jpg').resize((1000, 1333), Image.LANCZOS).save(FIG / 'ids_134905.jpg', quality=84)
Image.open(IDR / '20261030_135052_ids.jpg').crop((450, 500, 1450, 1300)).save(FIG / 'ids_135052_crop.jpg', quality=84)
shutil.copy(IDR / '20261030_134905_sheet.jpg', FIG / 'ids_134905_sheet.jpg')
e_ = Image.open(ESPR / (Path(ESPF[0]['photo']).stem + '_ids.jpg')); e_.resize((1000, round(e_.height * 1000 / e_.width)), Image.LANCZOS).save(FIG / 'ids_esp.jpg', quality=84)
tot = sum(int(r['total']) for r in SM); stn = {k: sum(int(r[k]) for r in SM) for k in STAGES}; det_s = sorted(float(r['detect_s']) for r in SM[1:]); cls_ms = sum(float(r['classify_s']) for r in SM) / tot * 1000; assert tot == CK['flowers']
ok = lambda v: '일치' if v else '**불일치**'
T_ID = (['분할', '사진', '꽃'], [['val', sum(1 for m in read(MANIFEST) if m['split'] == 'val' and m['photo'] in FL), CK['val']['flowers']], ['test', sum(1 for m in read(MANIFEST) if m['split'] == 'test' and m['photo'] in FL), CK['test']['flowers']], ['합계', CK['photos'], tot]], (1, 2))
T_CK = (['확인 항목', '결과'], [['val 혼동 행렬과 기존 평가', ok(CK['val']['confusion_same'])], ['test 혼동 행렬과 기존 평가', ok(CK['test']['confusion_same'])], ['사진별 단계 개수와 기존 평가', ok(CK['val']['counts_same'] and CK['test']['counts_same'])], ['박스, 다각형, 번호 (빠짐과 중복 포함)', ok(not CK['errors'])], ['넓이가 0인 마스크', f"{CK['area_zero']}개"], ['마스크가 겹치는 ID 쌍', f"{CK['overlap_pairs']}쌍 (절반 이상 겹침 {CK['overlap_ge50']}쌍)"]], ())
T_PROB = (['예측 단계', '단계가 맞은 꽃', '확률 중앙값', '단계가 틀린 꽃', '확률 중앙값 '], [[k, v['correct'][0], f3(v['correct'][2]), v['wrong'][0], f3(v['wrong'][2])] for k, v in CK['stage_prob'].items()], (1, 2, 3, 4))
nm, ns = len(MERGED), len(SPLIT); tag = lambda L: ', '.join(f'§{m[0][9:15]}§ {m[1]}번' for m in L)
esp_n = sum(int(r['total']) for r in ESP)
ID_WHAT = ['**순서:** 세그멘테이션으로 꽃을 찾고, 위에서 아래, 왼쪽에서 오른쪽 순서로 번호를 매긴 뒤, 번호별로 단계를 분류합니다. 잎 모델 §v7_leafid§와 같은 규칙입니다.', '**번호 형식:** §<사진>_F0001§. **사진 한 장 안에서만 유효**하며, 다른 사진의 같은 꽃과 이어지지 않습니다.', '**분류 입력:** 모델 박스의 긴 변 × 1.25 정사각형 크롭입니다. 마스크는 분류에 쓰지 않고, ID별 영역 정보로 남깁니다.', '**출력:** §<사진>_ids.jpg§, §<사진>_sheet.jpg§, §flowers.csv§, §flowers.json§, §summary.csv§입니다.', f"**단계별 개수:** 봉오리 {stn['봉오리']}, 개화중 {stn['개화중']}, 만개 {stn['만개']}입니다."]
ID_CK_NOTE = 'ID를 붙이면서 꽃이나 단계가 바뀌지 않았으므로 **성능은 7절의 값 그대로입니다.** 추론 코드를 §infer.py§로 옮긴 뒤 평가를 다시 돌려 val, test 예측이 이전과 같은 것도 확인했습니다.'
ID_ONE = [f'**합쳐짐 {nm}개:** 한 마스크 안에 라벨 중심이 2개 이상 들어간 ID입니다. 그림으로 봐도 꽃 두 송이가 합쳐져 있습니다 ({tag(MERGED)}). 중복 라벨 정리 전에는 8개였고, 5개가 중복 라벨 때문이었습니다.', f'**쪼개짐 의심 {ns}건:** 한 라벨 박스 안에 ID가 2개 이상 들어간 경우입니다. {tag([[SPLIT[0][0], SPLIT[0][2][0]]])}과 {SPLIT[0][2][1]}번으로, 한 송이가 잎에 가려 둘로 나뉜 것으로 보입니다 (Claude의 인상).', f'전체 {tot}개에 비하면 적지만 **하한으로 봐야 합니다.** 라벨 중심이 마스크 밖이면 합쳐짐으로 세지 않고, 라벨이 없는 꽃은 셀 수 없습니다. 놓친 꽃에는 애초에 ID가 없습니다.', f"**볼록도는 합쳐짐 기준이 되지 못합니다.** 잎 모델처럼 0.8 미만을 찾으면 {CK['low_solidity']}개인데, 그림으로 보면 대부분 잎에 가려진 한 송이입니다 (Claude의 인상)."]
ID_PROB = '**봉오리의 단계 확률은 신뢰도로 쓸 수 없습니다.** 맞은 경우와 틀린 경우의 값이 거의 같습니다. 분류 학습의 클래스 가중치와 라벨 스무딩(0.1) 영향으로 추정하지만 확인하지 않았습니다. 개화중은 차이가 있지만 틀린 예측도 높은 값이 많습니다. 라벨과 IoU 0.5 이상으로 맞은 꽃만 셌고, val과 test를 합친 값입니다.'
ID_TIME = f'**처리 시간 (V100):** 탐지는 사진당 중앙값 {det_s[len(det_s) // 2]:.2f}초, 분류는 꽃 하나당 약 {cls_ms:.0f}ms입니다. 사진 열기와 그림 저장은 뺐고, 탐지 값에서 모델을 처음 올리는 첫 사진도 뺐습니다. Jetson에서는 재지 않았습니다.'
ID_ESP = [f"폴더 입력 확인을 겸해 잎 평가용 ESP 카메라 사진 {len(ESP)}장을 돌렸습니다. {sum(1 for r in ESP if int(r['total']) == 0)}장은 0개, 1장은 봉오리 {esp_n}개입니다 (탐지 신뢰도 {ESPF[0]['conf']}).", '그 1개는 확대해도 흐려서 진짜 봉오리인지 알 수 없습니다. 6장을 축소해 보면 큰 꽃은 보이지 않지만, 작은 봉오리는 그 크기로 판단할 수 없습니다.', '**꽃이 찍힌 사진이 아니라서 이 카메라의 꽃 탐지 성능은 아직 알 수 없습니다.** 라벨도 없습니다.']
INFO = [['유형', '데이터 + 모델'], ['작업일', '2026-11-03 ~ 04'], ['비교 기준', 'v1_box §det/third_pseudo§ + §cls/effnetv2s_r2§'], ['모델', '§runs/seg/first§ (YOLO26s-seg, §flower§ 1클래스, 80회)'], ['상태', '**세그멘테이션 학습, 비교, ID 부여 완료. 최종 확정 전**']]
SUM = [f"박스 라벨과 임시 라벨 {N:,}개로 SAM2.1 마스크 초안을 만들어 세그멘테이션 모델을 학습했고, 찾은 꽃마다 ID를 붙여 단계를 분류합니다.", f"사용자가 v1 오류 시트 167칸, v2 오탐 시트 18칸, v2만 놓친 라벨 35칸을 판정해 test 라벨을 고쳤습니다. 지금 test F1은 v1 {f3(c1['f1'])}, v2 {f3(c2['f1'])}이고, 정밀도는 {f3(c1['precision'])}, {f3(c2['precision'])}, 재현율은 {f3(c1['recall'])}, {f3(c2['recall'])}입니다.", f"**봉오리를 세는 성능은 v1이 낫습니다.** 사진당 봉오리 개수 오차는 v1 {e1['count_mae_per_photo'][0]:.2f}, v2 {e2['count_mae_per_photo'][0]:.2f}입니다. v2는 오탐이 적고 큰 꽃을 더 찾습니다.", '**모델은 아직 정하지 않았습니다.** 한눈에 보는 표와 사진은 [비교 페이지](COMPARE.md)에 있습니다.', f'**ID 부여를 붙였습니다.** val, test {CK["photos"]}장의 꽃 {tot}개에 번호를 매겼고, 박스와 단계는 기존 평가와 같습니다. 번호는 사진 한 장 안에서만 유효합니다.']
METHOD = ['**대상:** 사진 133장의 모든 꽃. 봉오리, 개화중, 만개를 구분하지 않고 만듭니다.', '**창:** 박스 긴 변의 0.75배 + 16px 여백으로 잘라 SAM에 넣습니다. 사진 전체를 넣으면 작은 봉오리가 뭉개지기 때문입니다.', '**후처리:** 조각이 여러 개면 가장 큰 덩어리만 남기고 다각형으로 저장합니다 (허용 오차 1px).', '**임시 라벨:** v1 §det/second§의 신뢰도 0.7 이상 예측 중 기존 라벨, 제외 영역, 차단 목록과 겹치지 않는 3,280개입니다. 판정을 마친 표본 100개는 라벨 또는 제외 영역으로 옮겼습니다.', '**저장:** §labels/masks.jsonl§에 사진, 라벨 번호, 단계, 박스, 다각형, 품질 지표를 기록합니다. 중복 정리와 판정으로 지운 라벨의 마스크 18개를 빼 지금은 5,431개입니다.']
RETRY = [f'IoU 0.5 미만이거나 마스크가 없는 {fail}개는 창 크기와 프롬프트 박스 크기를 바꿔 다시 만들었습니다. **{fixed}개를 고쳤고 {failed}개는 학습에서 회색으로 가렸습니다.**', '기준을 넘겼어도 재생성 마스크의 일부는 옆 꽃이나 잎으로 번진 것처럼 보입니다. 전체의 2% 미만이라 그대로 두었습니다.', '0.5~0.7 구간은 대부분 쓸 만해 보여 그대로 썼고, 꽃받침이 빠진 마스크도 일괄 수정하지 않았습니다.']
LOOK = ['**대부분 윤곽을 잘 따라갑니다.** 봉오리, 만개, 초록 봉오리 모두 그렇습니다.', '**색 있는 꽃잎만 잡는 경우가 있습니다.** §K001§, §K003§, §X045§, §X048§처럼 초록 꽃받침이 빠집니다. 꽃받침을 포함하는 박스 라벨 규칙과 다릅니다.', '**거의 실패한 마스크가 있었습니다.** §K022§, §K056§, §X021§, §X035§는 일부만 잡았고, §K046§, §K087§, §X032§, §X068§은 비어 있거나 거의 비어 있었습니다 (재생성 전).', '**지표가 나빠도 멀쩡한 것이 많습니다.** 품질 지표에 걸린 573개 중 표본 70개를 보면 상당수는 쓸 만해 보였습니다.']
TRAIN = ['**구조:** YOLO26s-seg, 파라미터 약 1,037만 개(추론 기준), COCO 사전학습 §yolo26s-seg.pt§에서 시작했습니다.', '**v1과 같은 조건:** 사진, 타일과 축척, 증강, 회차(80회), 임시 라벨 출처가 §third_pseudo§와 같습니다. 라벨이 박스에서 다각형으로 바뀌었고, 표본 판정 100개가 반영된 점만 다릅니다.', f"**학습 데이터:** train 타일 1,404장, 대상 12,394개. V100에서 약 30분 걸렸습니다.", f"**타일 기준 val (SAM 초안 라벨, 임계값 없음):** mAP50 박스 {f3(best['metrics/mAP50(B)'])}, 마스크 {f3(best['metrics/mAP50(M)'])}. 사진 단위 점수와 다른 값입니다."]
NOTE_DET = [f"**v2의 낮은 임계값:** 임계값 0.55는 정밀도 {f3(D2['test']['sweep']['0.55']['precision'])}, 재현율 {f3(D2['test']['sweep']['0.55']['recall'])}입니다. test를 보고 고른 참고값이라 후보 선정에는 쓰지 않습니다.", f"**마스크를 감싼 박스:** test F1이 {f3(MB['test']['chosen']['f1'])}로 모델 박스의 {f3(c2['f1'])}보다 낮습니다. 마스크에서 꽃받침이 빠지는 것이 원인일 수 있지만 확인하지 않았습니다. 그래서 분류 입력과 박스 평가에는 모델 박스를 씁니다.", f"**놓침 비교:** 둘 다 놓침 {pr['both_miss']}개, v1만 놓침 {pr['only_v1_miss']}개, v2만 놓침 {pr['only_v2_miss']}개입니다. v2만 놓친 {pr['only_v2_miss']}개 중 {CP['stage']['new_miss'].get('봉오리', 0)}개가 봉오리입니다."]
NOTE_E2E = '매크로 F1 차이는 구간 안이어서 의미 있는 차이가 아닙니다. 봉오리 개수 오차는 v2가 나쁘고, 개화중과 만개 오차는 비슷합니다. 분류기는 v1과 같습니다. 이 평가에도 추가 라벨의 영향이 들어 있습니다.'
PH = {r[0]: r for r in CP['rows']}; PM1, PM2 = (next(r for r in c_['per_photo'] if '135052' in r['photo']) for c_ in (c1, c2))
PROG = ['**큰 꽃 놓침, 일부 해결:** §M008§, §M009§를 이제 찾습니다. 긴 변 1,830px인 §M024§는 여전히 못 찾습니다.', '**붙은 꽃, 일부 해결:** 분홍 꽃 두 송이(§F005§)는 따로 잡히지만, 빨간 꽃 두 송이(§F002§, §M002§, §M003§)는 아직 하나로 묶입니다. 번호는 사용자가 판정한 v1 시트 기준입니다.', f"**오탐 집중, 판정으로 해소:** 사진 §135052§의 v1 오탐 35개는 대부분 라벨이 빠진 실제 봉오리였습니다. 지금 오탐은 v1 {PM1['fp']}개, v2 {PM2['fp']}개입니다.", f"**새로 생긴 약점:** v2는 먼 봉오리가 몰린 곳을 더 놓칩니다. 놓침이 사진 §140440§은 v1 {PH['140440'][6]}개, v2 {PH['140440'][7]}개이고, §140433§은 {PH['140433'][6]}개, {PH['140433'][7]}개입니다.", '**개화중 분류:** 그대로 약합니다. 세그멘테이션으로 해결되는 문제가 아닙니다.']
CMP = J(V2R / 'compare/v1_v2.json')
LIMITS = ['**추가 라벨의 박스는 대부분 v1의 예측입니다.** 그래서 비교가 v1에 조금 유리합니다. 추가 라벨을 뺀 값을 함께 봐야 합니다.', '**학습은 라벨 정리 전 데이터로 했습니다.** v1, v2 모두 train의 중복 4쌍이 든 채로 학습했고, 재학습하지 않았습니다. test 판정은 학습 데이터에 영향이 없고, 평가만 최신 라벨로 다시 계산했습니다.', '**마스크 라벨은 SAM 초안입니다.** 사람이 그린 정답이 없어 마스크 정확도를 평가하지 못했고, 박스 기준으로만 비교했습니다.', '**품질 지표는 정답과의 비교가 아닙니다.** 프롬프트 박스와 맞는지 보는 값이라, 박스 안에서 일부만 잡아도 박스가 비슷하면 높게 나옵니다.', '**임시 라벨 위의 마스크입니다.** 표본 판정에서 임시 라벨의 약 96%가 꽃이나 봉오리였지만, 나머지 오류는 마스크에도 그대로 남습니다.', '**test는 화분 단위로 분리되지 않았고, 휴대폰 사진 기준입니다.** ESP32 사진 평가가 없습니다.', '**ID는 사진 안에서만 유효합니다.** 같은 꽃을 시간에 따라 따라갈 수 없습니다.', '**Jetson 추론 시간을 재지 않았습니다.** V100 값만 있습니다.', '**학습은 한 번뿐입니다.** 씨앗을 바꾼 반복 학습이 없어 작은 차이는 우연일 수 있습니다.']
NEXT = ['쓸 모델 결정 (v2 재현율 개선 또는 v1에 박스 ID 부여)', '봉오리 놓침 원인 확인 (먼 봉오리, 임계값)', '개화중 분류 개선', '꽃이 찍힌 ESP32 사진으로 독립 평가', '같은 꽃을 여러 사진에서 잇는 추적 (시간에 따른 변화를 볼 때 필요)']
FILES = [['§labels/masks.jsonl§', '마스크 초안 (재생성 반영)'], ['§scripts/10_make_masks.py§', '마스크 생성, 재생성'], ['§scripts/20_export_seg.py§', '학습 데이터 내보내기'], ['§scripts/30_train_seg.py§', '학습'], ['§scripts/31_eval_seg.py§', '사진 단위 평가'], ['§scripts/32_compare_v1.py§', 'v1 비교'], ['§COMPARE.html§, §COMPARE.md§', 'v1, v2 비교 페이지'], ['§scripts/33_compare_page.py§', '비교 페이지 생성'], ['§scripts/35_unjudged_sheet.py§', 'v2만 놓친 미판정 라벨 시트'], ['§scripts/34_time_models.py§', '추론 시간 측정'], ['§scripts/13_drop_masks.py§', '중복 라벨의 마스크 제거'], ['§scripts/infer.py§', '탐지와 분류 추론 코드'], ['§scripts/40_flower_ids.py§', 'ID 부여와 ID별 분류'], ['§scripts/41_check_ids.py§', 'ID 결과 확인 (§check.json§)'], ['§runs/ids/§', 'ID 결과 (커밋하지 않음)'], ['§runs/seg/first/§', '모델과 평가 결과 (커밋하지 않음)'], ['§runs/mask_check/§', '확인 시트 원본 (커밋하지 않음)'], ['§../v1_box/labels/§', '박스 라벨 원본']]


def inline(s):
    s = re.sub(r'§([^§]+)§', r'<code>\1</code>', s); s = re.sub(r'\[([^]]+)\]\(([^)]+)\.md\)', r'<a href="\2.html">\1</a>', s)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)


def esc(c):
    return inline(str(c).replace('<', '&lt;').replace('>=', '≥'))


def html_table(head, rows_, num=()):
    th = ''.join(f'<th class="n">{inline(h)}</th>' if i in num else f'<th>{inline(h)}</th>' for i, h in enumerate(head))
    tr = ''.join('<tr>' + ''.join(f'<td class="n">{esc(c)}</td>' if i in num else f'<td>{esc(c)}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows_)
    return f'<div class="tbl"><table><tr>{th}</tr>{tr}</table></div>'


def md_table(head, rows_, num=()):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---:' if i in num else '---' for i in range(len(head))) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows_])


uri = lambda n: 'data:image/jpeg;base64,' + base64.b64encode((FIG / n).read_bytes()).decode()
plain = lambda s: s.replace('§', '')
ul = lambda it: '<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in it) + '</ul>'
sec = lambda n, t, b: f'<section><h2><span class="num">{n:02d}</span>{t}</h2>{b}</section>'
fig = lambda n, cap: f'<figure><img src="{uri(n)}" alt="{plain(cap)}"><figcaption>{inline(cap)}</figcaption></figure>'
kp = lambda a, b: f'<div class="kpi"><b>{a}</b><span>{b}</span></div>'
arrow = lambda a, b: f'<s>{a} →</s> {b}'
flow = '<div class="flow">' + '<div class="arrow">→</div>'.join(f'<div class="step{" key" if k else ""}"><b>{a}</b><small>{b}</small></div>' for a, b, k in [('박스 라벨', 'v1 라벨 + 임시 라벨', 0), ('SAM2.1', '박스를 프롬프트로 마스크 초안', 1), ('재생성', '실패 마스크', 0), ('YOLO26s-seg', 'flower 1클래스 학습', 1), ('합치기', '타일 결과를 사진 좌표로', 0), ('ID', '사진 안 번호', 1), ('분류', 'ID별, 모델 박스 크롭', 0)]) + '</div>'
STYLE = '<style>' + (V1 / 'report/style.css').read_text(encoding='utf-8') + 'td code,li code{word-break:normal;white-space:nowrap}</style>'
TITLE = '꽃 모델 v2 리포트: 마스크 라벨, 세그멘테이션, ID 부여'
LEG = '초록 맞게 찾음, 빨강 오탐, 파랑 놓침'
RV = [(a, json.loads((V1 / 'labels' / f'{b}.json').read_text(encoding='utf-8'))) for a, b in [('test 1차', 'review_test_fp'), ('test 1차', 'review_test_fn'), ('val', 'review_val_fp'), ('val', 'review_val_fn'), ('train 13장', 'review_train_fp'), ('임시 라벨 표본', 'review_pseudo_sample_r1'), ('test 2차', 'review_test_fp_r2'), ('test 2차', 'review_test_fn_r2'), ('test 3차', 'review_test_fp_r3'), ('test 4차', 'review_test_u_r4')]]; RV_N = sum(len(d) for _, d in RV)
LAB_ITEMS = ['**과정:** Claude가 초안을 만들고, 모델이 틀린 곳을 번호 칸으로 잘라 Claude가 먼저 확인한 뒤, **사용자가 칸마다 판정합니다.** Claude의 인상으로는 라벨을 바꾸지 않습니다.', f'**사용자가 판정한 칸:** 지금까지 {RV_N}개입니다. test 17장은 사진 단위 검수도 했습니다. 회차별 내역과 칸별 판정은 [v1 리포트](../v1_box/REPORT.md)의 라벨링 절에 있습니다.', '**v2에 해당하는 부분:** 박스 라벨은 v1과 같은 원본을 씁니다. **SAM 마스크는 사람 판정이 없습니다.** v2의 오탐 18칸과 v2만 놓친 라벨 35칸은 판정했습니다.', '**판정으로 추가한 라벨 44개는 마스크가 없습니다.** test 라벨이라 학습에는 쓰지 않습니다.']
LAB_HTML = ul([re.sub(r'\[([^]]+)\]\(([^)]+)\)', lambda m: '<a href="../v1_box/REPORT.html">' + m[1] + '</a>', x) for x in LAB_ITEMS])
ID_HTML = ul(ID_WHAT) + html_table(*T_ID) + fig('ids_134905.jpg', '사진 §134905§의 ID 결과. 초록은 봉오리, 노랑은 개화중, 분홍은 만개입니다.') + fig('ids_135052_crop.jpg', '봉오리가 많은 사진 §135052§ 일부. 번호보다 작은 꽃은 번호를 박스 위에 씁니다.') + fig('ids_134905_sheet.jpg', '사진 §134905§의 ID별 시트. 칸 위에 번호, 단계, 단계 확률을 적었습니다.') + '<h3>기존 평가와 일치 확인</h3>' + html_table(*T_CK) + '<p>' + inline(ID_CK_NOTE) + '</p><h3>ID 하나가 꽃 하나인가</h3>' + ul(ID_ONE) + fig('id_merged_split.jpg', '합쳐짐, 쪼개짐 의심 전부. 하늘색은 예측 마스크, 파랑은 라벨 박스입니다.') + fig('id_low_solidity.jpg', '볼록도 0.8 미만 전부') + '<h3>단계 확률</h3>' + html_table(*T_PROB) + '<div class="note">' + inline(ID_PROB) + '</div><h3>처리 시간</h3><p>' + inline(ID_TIME) + '</p><h3>ESP 카메라 사진</h3>' + ul(ID_ESP) + fig('ids_esp.jpg', 'ESP 카메라 사진에서 나온 유일한 예측. 정답 라벨은 없습니다.')
ID_MD = [f'- {x}' for x in ID_WHAT] + ['', md_table(*T_ID), '', '![사진 134905의 ID 결과](report/ids_134905.jpg)', '', '### 기존 평가와 일치 확인', md_table(*T_CK), '', ID_CK_NOTE, '', '### ID 하나가 꽃 하나인가'] + [f'- {x}' for x in ID_ONE] + ['', '![합쳐짐, 쪼개짐 의심](report/id_merged_split.jpg)', '', '### 단계 확률', md_table(*T_PROB), '', ID_PROB, '', '### 처리 시간', ID_TIME, '', '### ESP 카메라 사진'] + [f'- {x}' for x in ID_ESP]
body = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 v2 리포트</title>{STYLE}</head><body><div class="wrap"><header><div class="eyebrow">Helpai · 스마트큐브 꽃 상태 확인 · 2026-11-03 ~ 04</div><h1>{TITLE}</h1><p class="lead">v1의 박스 탐지를 세그멘테이션으로 바꿨습니다. 오탐은 줄고 큰 꽃을 더 찾지만, 봉오리를 더 놓칩니다. 찾은 꽃마다 번호를 매겨 번호별로 단계를 분류합니다.</p><div class="meta"><span>비교 기준: {inline('v1_box §third_pseudo§')}</span><span>평가: 휴대폰 사진, 사진 단위, IoU 0.5</span><span>최종 확정 전</span></div><div class="nav"><a href="../README.md">꽃 모델 버전 목록</a><a href="COMPARE.html">v1, v2 비교</a><a href="../v1_box/REPORT.html">← v1_box</a></div></header>
{sec(0, '요약', '<div class="kpis">' + kp(arrow(f3(c1['f1']), f3(c2['f1'])), 'test 탐지 F1') + kp(arrow(f3(c1['precision']), f3(c2['precision'])), 'test 정밀도') + kp(arrow(f3(c1['recall']), f3(c2['recall'])), 'test 재현율') + kp(arrow(f"{e1['count_mae_per_photo'][0]:.2f}", f"{e2['count_mae_per_photo'][0]:.2f}"), '사진당 봉오리 개수 오차 (나빠짐)') + '</div>' + ul(SUM))}
{sec(1, '마스크 라벨 만들기', flow + ul(METHOD))}
{sec(2, '마스크 품질 (생성 직후)', html_table(*T_ALL) + '<h3>단계별</h3>' + html_table(*T_STAGE) + '<h3>크기별</h3>' + html_table(*T_SIZE) + '<h3>분할별</h3>' + html_table(*T_SPLIT) + '<h3>출처별</h3>' + html_table(*T_SRC) + '<h3>재생성</h3>' + ul(RETRY))}
{sec(3, '마스크 시트를 본 인상', '<div class="note">Claude가 썸네일로 본 인상입니다. 사람 판정이 아닙니다.</div>' + ul(LOOK) + fig('sample_top.jpg', '단계별 표본 시트 윗부분 (재생성 전). 노란 선이 마스크, 파란 선이 프롬프트 박스, 칸 아래 숫자가 IoU입니다.') + fig('flagged_top.jpg', '품질 지표에 걸린 마스크 시트 윗부분 (재생성 전)'))}
{sec(4, '라벨링: Claude 검수 후 사람 판정', LAB_HTML)}
{sec(5, '세그멘테이션 학습', ul(TRAIN))}
{sec(6, 'v1과 탐지 비교', html_table(*T_DET) + '<p>' + inline('표와 사진을 나란히 보려면 §COMPARE.html§을 엽니다.') + '</p><h3>라벨 정리에 따른 변화</h3><p>' + inline(DUP_NOTE) + '</p>' + html_table(*T_DUP) + '<h3>판정 편향 확인</h3>' + html_table(*T_FAIR) + ul(FAIR_NOTE) + '<h3>모델 선택</h3>' + ul(DEC) + '<h3>구분별 재현율</h3>' + html_table(*T_BY) + ul(NOTE_DET) + '<h3>사진별</h3><p>저장된 예측값의 반올림 때문에 v1 오탐 합계가 위 표보다 1개 적습니다.</p>' + html_table(*T_PHOTO))}
{sec(7, '탐지와 분류 연결', html_table(*T_E2E) + html_table(*T_E3) + '<h3>' + 'v2 test 혼동 행렬' + '</h3>' + html_table(*T_E4) + '<div class="note">' + inline(NOTE_E2E) + '</div>')}
{sec(8, 'ID 부여와 ID별 분류', ID_HTML)}
{sec(9, 'v1 평가에서 본 문제의 변화', ul(PROG) + '<div class="figs">' + fig('v2_134905.jpg', f'v2, 사진 §134905§: {LEG}') + fig('v2_140440.jpg', 'v2, 사진 §140440§') + '</div>' + fig('v2_135052.jpg', 'v2, 사진 §135052§'))}
{sec(10, '한계', ul(LIMITS))}
{sec(11, '다음 단계', '<ol>' + ''.join(f'<li>{inline(x)}</li>' for x in NEXT) + '</ol>')}
{sec(12, '파일', html_table(['경로', '내용'], FILES))}
<footer>표의 수치는 scripts/12_report.py가 평가 결과 파일과 마스크 파일에서 읽어 만들었습니다.</footer></div></body></html>'''
(VER / 'REPORT.html').write_text(body, encoding='utf-8')
md = [f'# {TITLE}', '', '> 그림을 포함한 리포트: [REPORT.html](REPORT.html) (비교 기준: [v1_box](../v1_box/REPORT.md))', '', md_table(['항목', '내용'], INFO), '', '## 요약'] + [f'- {x}' for x in SUM] + ['', '## 마스크 라벨 만들기'] + [f'- {x}' for x in METHOD] + ['', '## 마스크 품질 (생성 직후)', md_table(*T_ALL), '', '### 단계별', md_table(*T_STAGE), '', '### 크기별', md_table(*T_SIZE), '', '### 분할별', md_table(*T_SPLIT), '', '### 출처별', md_table(*T_SRC), '', '### 재생성'] + [f'- {x}' for x in RETRY] + ['', '## 마스크 시트를 본 인상', 'Claude가 썸네일로 본 인상입니다. 사람 판정이 아닙니다.', ''] + [f'- {x}' for x in LOOK] + ['', '![단계별 표본 시트](report/sample_top.jpg)', '', '## 라벨링: Claude 검수 후 사람 판정'] + [f'- {x}' for x in LAB_ITEMS] + ['', '## 세그멘테이션 학습'] + [f'- {x}' for x in TRAIN] + ['', '## v1과 탐지 비교', md_table(*T_DET), '', '### 라벨 정리에 따른 변화', DUP_NOTE, '', md_table(*T_DUP), '', '### 판정 편향 확인', md_table(*T_FAIR), ''] + [f'- {x}' for x in FAIR_NOTE] + ['', '### 모델 선택'] + [f'- {x}' for x in DEC] + ['', '### 구분별 재현율', md_table(*T_BY), ''] + [f'- {x}' for x in NOTE_DET] + ['', '### 사진별', '저장된 예측값의 반올림 때문에 v1 오탐 합계가 위 표보다 1개 적습니다.', '', md_table(*T_PHOTO), '', '## 탐지와 분류 연결', md_table(*T_E2E), '', md_table(*T_E3), '', '### v2 test 혼동 행렬', md_table(*T_E4), '', NOTE_E2E, '', '## ID 부여와 ID별 분류'] + ID_MD + ['', '## v1 평가에서 본 문제의 변화'] + [f'- {x}' for x in PROG] + ['', f'![v2, 134905: {LEG}](report/v2_134905.jpg)', '', '## 한계'] + [f'- {x}' for x in LIMITS] + ['', '## 다음 단계'] + [f'{i + 1}. {x}' for i, x in enumerate(NEXT)] + ['', '## 파일', md_table(['경로', '내용'], FILES), '']
(VER / 'REPORT.md').write_text('\n'.join(md).replace('§', chr(96)), encoding='utf-8')
print('html', (VER / 'REPORT.html').stat().st_size, 'md', (VER / 'REPORT.md').stat().st_size, 'leftover', body.count('§'), dict(cnt), 'fixed', fixed, 'failed', failed)
