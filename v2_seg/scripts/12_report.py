# v2 마스크 단계 리포트를 만든다. 수치는 labels/masks.jsonl에서 읽고, 그림은 report/에 저장해 REPORT.html에 넣는다. 본문의 §는 코드 표시다.
# 사용: python 12_report.py
import base64, collections as C, re
from common import *
from PIL import Image

R = VER / 'runs' / 'mask_check'; FIG = VER / 'report'; FIG.mkdir(exist_ok=True); rows = read_masks()
name = lambda r: r['stage'] or '임시 라벨'; N = len(rows)
SEV = ['IoU 0.7 이상', '0.5~0.7', '0.3~0.5', '0.3 미만', '마스크 없음']


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


cnt = C.Counter(sev(r) for r in rows); fail = cnt[SEV[2]] + cnt[SEV[3]] + cnt[SEV[4]]; assert sum(cnt.values()) == N
T_ALL = (['마스크 박스와 프롬프트 박스의 IoU', '개수', '비율'], [[s, cnt[s], f'{cnt[s] / N * 100:.1f}%'] for s in SEV], (1, 2))
T_STAGE = table_by(name, STAGES + ['임시 라벨']); T_SIZE = table_by(size, ['40px 미만', '40~80px', '80~160px', '160px 이상']); T_SPLIT = table_by(lambda r: r['split'], ['train', 'val', 'test']); T_SRC = table_by(lambda r: r['source'], ['draft', 'legacy', 'review', 'pseudo'])
for n_, src, h in [('sample_top.jpg', 'sample', 750), ('flagged_top.jpg', 'flagged', 750)]:
    s_ = Image.open(R / src / 'sheet_masks.jpg'); s_.crop((0, 0, s_.width, h)).save(FIG / n_, quality=84)
INFO = [['유형', '데이터 (학습 전)'], ['작업일', '2026-11-03'], ['입력', 'v1 박스 라벨 + §third_pseudo§와 같은 출처의 임시 라벨'], ['도구', 'SAM2.1 small (§sam2.1_s.pt§)'], ['상태', '**마스크 초안 생성 완료. 재생성, 학습, ID 부여는 시작 전**']]
SUM = [f'박스 라벨과 임시 라벨 {N:,}개를 프롬프트로 꽃 마스크 초안을 만들었습니다. V100에서 약 7분 걸렸습니다.', f'마스크를 감싼 박스가 프롬프트 박스와 IoU 0.7 이상으로 맞는 것이 {cnt[SEV[0]] / N * 100:.1f}%입니다.', f'IoU 0.5 미만이거나 마스크가 없는 {fail}개({fail / N * 100:.1f}%)는 그대로 쓸 수 없습니다.', '**사람이 전부 보지 않은 초안입니다.** Claude가 시트 170칸을 썸네일로 봤을 뿐이고, 사용자 판정은 아직 없습니다.']
METHOD = ['**대상:** 사진 133장의 모든 꽃. 봉오리, 개화중, 만개를 구분하지 않고 만듭니다.', '**창:** 박스 긴 변의 0.75배 + 16px 여백으로 잘라 SAM에 넣습니다. 사진 전체를 넣으면 작은 봉오리가 뭉개지기 때문입니다.', '**후처리:** 조각이 여러 개면 가장 큰 덩어리만 남기고 다각형으로 저장합니다 (허용 오차 1px).', '**임시 라벨:** v1 §det/second§의 신뢰도 0.7 이상 예측 중 기존 라벨, 제외 영역, 차단 목록과 겹치지 않는 3,280개입니다. 판정을 마친 표본 100개는 라벨 또는 제외 영역으로 옮겼습니다.', '**저장:** §labels/masks.jsonl§에 사진, 라벨 번호, 단계, 박스, 다각형, 품질 지표를 기록합니다.']
LOOK = ['**대부분 윤곽을 잘 따라갑니다.** 봉오리, 만개, 초록 봉오리 모두 그렇습니다.', '**색 있는 꽃잎만 잡는 경우가 있습니다.** §K001§, §K003§, §X045§, §X048§처럼 초록 꽃받침이 빠집니다. 꽃받침을 포함하는 박스 라벨 규칙과 다릅니다.', '**거의 실패한 마스크가 있습니다.** §K022§, §K056§, §X021§, §X035§는 일부만 잡았고, §K046§, §K087§, §X032§, §X068§은 비어 있거나 거의 비어 있습니다.', '**지표가 나빠도 멀쩡한 것이 많습니다.** 품질 지표에 걸린 573개 중 표본 70개를 보면 상당수는 쓸 만해 보였습니다. 지표만으로 지우면 좋은 마스크도 함께 버립니다.']
PLAN = [f'**IoU 0.5 미만과 마스크 없음 ({fail}개):** 창 크기를 바꿔 다시 생성하고, 그래도 실패하면 학습에서 회색으로 가립니다. 라벨 없이 두면 그 꽃이 배경으로 학습되기 때문입니다.', f'**0.5~0.7 ({cnt[SEV[1]]}개):** 대부분 쓸 만해 보여 그대로 씁니다.', '**꽃받침 빠짐:** 일괄 수정하지 않고 학습 결과에서 영향을 확인합니다.']
LIMITS = ['**품질 지표는 정답과의 비교가 아닙니다.** 프롬프트 박스와 맞는지 보는 값이라, 박스가 틀리면 마스크가 맞아도 낮게 나옵니다. 박스 안에서 일부만 잡아도 박스가 비슷하면 높게 나올 수 있습니다.', '**val, test의 마스크도 초안입니다.** 마스크 IoU로 낸 점수는 사람이 그린 정답과의 비교가 아닙니다. v1과의 비교는 박스 기준으로 합니다.', '**임시 라벨 위의 마스크입니다.** 표본 판정에서 임시 라벨의 약 96%가 꽃이나 봉오리였지만, 나머지 오류는 마스크에도 그대로 남습니다.', '**ESP32 사진은 포함되지 않습니다.**']
NEXT = ['실패 마스크 재생성과 학습 데이터 내보내기', '세그멘테이션 학습 (§flower§ 1클래스)', '사진 안 ID 부여와 ID별 분류 연결', 'v1과 비교: 탐지 점수, 단계별 재현율, 큰 꽃 재현율, 붙은 꽃 분리, 사진당 개수 오차']
FILES = [['§labels/masks.jsonl§', '마스크 초안'], ['§scripts/10_make_masks.py§', '마스크 생성'], ['§scripts/11_mask_sheet.py§', '확인 시트 생성'], ['§runs/mask_check/§', '확인 시트 원본 (커밋하지 않음)'], ['§../v1_box/labels/§', '박스 라벨 원본']]


def inline(s):
    s = re.sub(r'§([^§]+)§', r'<code>\1</code>', s)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)


def html_table(head, rows_, num=()):
    th = ''.join(f'<th class="n">{inline(h)}</th>' if i in num else f'<th>{inline(h)}</th>' for i, h in enumerate(head))
    tr = ''.join('<tr>' + ''.join(f'<td class="n">{inline(str(c))}</td>' if i in num else f'<td>{inline(str(c))}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows_)
    return f'<div class="tbl"><table><tr>{th}</tr>{tr}</table></div>'


def md_table(head, rows_, num=()):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---:' if i in num else '---' for i in range(len(head))) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows_])


uri = lambda n: 'data:image/jpeg;base64,' + base64.b64encode((FIG / n).read_bytes()).decode()
plain = lambda s: s.replace('§', '')
ul = lambda it: '<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in it) + '</ul>'
sec = lambda n, t, b: f'<section><h2><span class="num">{n:02d}</span>{t}</h2>{b}</section>'
fig = lambda n, cap: f'<figure><img src="{uri(n)}" alt="{plain(cap)}"><figcaption>{inline(cap)}</figcaption></figure>'
kp = lambda a, b: f'<div class="kpi"><b>{a}</b><span>{b}</span></div>'
flow = '<div class="flow">' + '<div class="arrow">→</div>'.join(f'<div class="step{" key" if k else ""}"><b>{a}</b><small>{b}</small></div>' for a, b, k in [('박스 라벨', 'v1 라벨 + 임시 라벨', 0), ('창 자르기', '박스 주변', 0), ('SAM2.1', '박스를 프롬프트로', 1), ('후처리', '가장 큰 덩어리, 다각형', 0), ('품질 지표', '박스 IoU, 채움, 벗어남', 0), ('확인 시트', '단계별 표본', 1)]) + '</div>'
STYLE = '<style>' + (FLOWER / 'v1_box/report/style.css').read_text(encoding='utf-8') + 'td code{word-break:normal;white-space:nowrap}</style>'
TITLE = '꽃 모델 v2 리포트 1: 마스크 라벨'
body = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 v2 리포트</title>{STYLE}</head><body><div class="wrap"><header><div class="eyebrow">Helpai · 스마트큐브 꽃 상태 확인 · 2026-11-03</div><h1>{TITLE}</h1><p class="lead">v1의 박스 탐지를 세그멘테이션으로 바꾸기 위해 박스 라벨에서 마스크 초안을 만들었습니다. 아직 학습 전이라 모델 점수는 없습니다.</p><div class="meta"><span>비교 기준: v1_box (학습 후 비교)</span><span>사용자 판정 없음</span></div><div class="nav"><a href="../README.md">꽃 모델 버전 목록</a><a href="../v1_box/REPORT.html">← v1_box</a></div></header>
{sec(0, '요약', '<div class="kpis">' + kp(f'{N:,}', '만든 마스크 대상') + kp(f'{cnt[SEV[0]] / N * 100:.1f}%', 'IoU 0.7 이상') + kp(f'{fail / N * 100:.1f}%', f'0.5 미만 또는 없음 ({fail}개)') + kp('0칸', '사용자 판정') + '</div>' + ul(SUM))}
{sec(1, '방법', flow + ul(METHOD))}
{sec(2, '품질 지표', html_table(*T_ALL) + '<h3>단계별</h3>' + html_table(*T_STAGE) + '<h3>크기별</h3>' + html_table(*T_SIZE) + '<h3>분할별</h3>' + html_table(*T_SPLIT) + '<h3>출처별</h3>' + html_table(*T_SRC))}
{sec(3, '시트를 본 인상', '<div class="note">Claude가 썸네일로 본 인상입니다. 사람 판정이 아닙니다.</div>' + ul(LOOK) + fig('sample_top.jpg', '단계별 표본 시트 윗부분. 노란 선이 마스크, 파란 선이 프롬프트 박스, 칸 아래 숫자가 IoU입니다.') + fig('flagged_top.jpg', '품질 지표에 걸린 마스크 시트 윗부분'))}
{sec(4, '학습 전 처리 (제안, 아직 적용하지 않음)', ul(PLAN))}
{sec(5, '한계', ul(LIMITS))}
{sec(6, '다음 단계', '<ol>' + ''.join(f'<li>{inline(x)}</li>' for x in NEXT) + '</ol>')}
{sec(7, '파일', html_table(['경로', '내용'], FILES))}
<footer>표의 수치는 scripts/12_report.py가 labels/masks.jsonl에서 읽어 만들었습니다.</footer></div></body></html>'''
(VER / 'REPORT.html').write_text(body, encoding='utf-8')
md = [f'# {TITLE}', '', '> 그림을 포함한 리포트: [REPORT.html](REPORT.html) (비교 기준: [v1_box](../v1_box/REPORT.md))', '', md_table(['항목', '내용'], INFO), '', '## 요약'] + [f'- {x}' for x in SUM] + ['', '## 방법'] + [f'- {x}' for x in METHOD] + ['', '## 품질 지표', md_table(*T_ALL), '', '### 단계별', md_table(*T_STAGE), '', '### 크기별', md_table(*T_SIZE), '', '### 분할별', md_table(*T_SPLIT), '', '### 출처별', md_table(*T_SRC), '', '## 시트를 본 인상', 'Claude가 썸네일로 본 인상입니다. 사람 판정이 아닙니다.', ''] + [f'- {x}' for x in LOOK] + ['', '![단계별 표본 시트](report/sample_top.jpg)', '', '## 학습 전 처리 (제안, 아직 적용하지 않음)'] + [f'- {x}' for x in PLAN] + ['', '## 한계'] + [f'- {x}' for x in LIMITS] + ['', '## 다음 단계'] + [f'{i + 1}. {x}' for i, x in enumerate(NEXT)] + ['', '## 파일', md_table(['경로', '내용'], FILES), '']
(VER / 'REPORT.md').write_text('\n'.join(md).replace('§', chr(96)), encoding='utf-8')
print('html', (VER / 'REPORT.html').stat().st_size, 'md', (VER / 'REPORT.md').stat().st_size, 'leftover', body.count('§'), dict(cnt))
