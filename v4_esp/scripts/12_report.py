# v4 리포트를 만든다: REPORT.html(그림 포함), REPORT.md(요약). 수치는 runs/eval의 결과 파일에서 읽는다. 사용: python 12_report.py
import base64, html, re
from common import *
J = lambda n: json.loads((VER / 'runs/eval' / n).read_text(encoding='utf-8')); T, Ph, Sel, Pred = J('test_result.json'), J('phone_val.json'), J('selected.json'), J('preds_test.json')
LAB = {}
for t in (VER / 'labels/masks_body.jsonl').read_text(encoding='utf-8').splitlines():
    r = json.loads(t); LAB.setdefault(r['photo'], []).append(r)
SP = splits(); N = {k: sum(v == k for v in SP.values()) for k in ('train', 'val', 'test', 'excluded')}


def count(name, t):
    f = o = 0
    for p, Q in Pred[name].items():
        u = set()
        for q in Q:
            if q[4] < t:
                continue
            cx, cy = (q[0] + q[2]) / 2, (q[1] + q[3]) / 2; i = next((j for j, r in enumerate(LAB[p]) if r['box'][0] <= cx <= r['box'][2] and r['box'][1] <= cy <= r['box'][3]), None)
            if i is None:
                o += 1
            elif i not in u:
                u.add(i); f += 1
    return f, o


NM = {'v3_t1280': 'v3, 타일 1280', 'v3_t880': 'v3, 타일 880', 'ft_s0_t880': '파인튜닝 시드 0, 타일 880', 'ft_s1_t880': '파인튜닝 시드 1, 타일 880', 'ft_s2_t880': '파인튜닝 시드 2, 타일 880'}
BUD = lambda r, b: '{}/{}'.format(*r['buds'][b])
rows_t = '\n'.join(f"| {NM[k]} | {r['thr']:.2f} | {r['found']} / {r['total']} | {r['other']} | {r['iou50']} | {BUD(r, 'P')} | {BUD(r, 'R')} | {BUD(r, 'G')} |" for k, r in T.items())
TH = (.2, .3, .4, .5, .6, .7)
rows_s = '\n'.join(f"| {NM[k]} | " + ' | '.join('{} ({})'.format(*count(k, t)) for t in TH) + ' |' for k in T)
rows_p = '\n'.join(f"| {k} | {v['box_map50']:.3f} | {v['box_p']:.3f} | {v['box_r']:.3f} | {v['mask_map50']:.3f} |" for k, v in Ph.items())
MD = f'''# 꽃 모델 v4 리포트: ESP 사진 파인튜닝

> 그림을 포함한 리포트: [REPORT.html](REPORT.html)

## 요약

- **사람이 라벨을 그린 ESP 사진으로 v3를 파인튜닝했습니다.** 사진 17장 중 학습 {N['train']}장, 검증 {N['val']}장, 평가 {N['test']}장을 쓰고, 손이 찍힌 {N['excluded']}장은 뺐습니다.
- **평가 사진에서 찾은 횟수가 늘었습니다.** 9번 중 v3 타일 1280은 2번, 타일 880은 0번, 파인튜닝은 시드에 따라 5, 6, 5번입니다. 다른 예측은 모두 0개입니다.
- **타일 880의 효과는 재현되지 않았습니다.** 이번 평가에서는 임계값 0.60에서 타일 880이 1280보다 낮고, 0.50에서는 같습니다.
- **휴대폰 성능은 떨어졌습니다.** 휴대폰 val 박스 mAP50이 {Ph['v3']['box_map50']:.2f}에서 {min(Ph[f'ft_s{s}']['box_map50'] for s in range(3)):.2f}~{max(Ph[f'ft_s{s}']['box_map50'] for s in range(3)):.2f}로 내려갔습니다. 이 가중치는 ESP 전용입니다.
- **새 봉오리에 대한 평가가 아닙니다.** 평가의 봉오리 3개는 모두 학습 사진에도 있어, 배치가 바뀌어도 같은 봉오리를 찾는다는 것까지만 말할 수 있습니다.
- **cameras.json과 v3는 바꾸지 않았습니다.** 어떤 모델을 쓸지도 아직 정하지 않았습니다.

## 데이터

| 용도 | 사진 | 내용 |
|---|---:|---|
| train | {N['train']} | F-6의 6장, 왼쪽 화분이 잘 보이는 배치 5장 |
| val | {N['val']} | 260eac, 804b80 |
| test | {N['test']} | 6964dd, 8583c9, eee390. 잎이 왼쪽을 덮은 배치 |
| 제외 | {N['excluded']} | 1859c6. 손이 찍혔고 평가와 배치가 겹칩니다 |

분리는 라벨을 그리기 전에 정했고, 1859c6만 학습 전 검수에서 뺐습니다. 라벨 페이지는 사진 순서를 섞고 용도를 보여 주지 않았습니다. 사람이 그린 상자는 47개이고 모두 봉오리입니다. 사진은 같은 트레이에서 찍었습니다. EXIF가 없어 촬영 시각은 확인하지 못했습니다.

## 라벨

사람이 그린 상자는 꽃받침과 여백을 포함해 모델 상자보다 컸습니다. F-6 사진의 봉오리 A에서 모델 상자와의 IoU는 0.26~0.40입니다. 라벨 페이지 안내에 꽃받침을 포함하라고 적은 것이 원인입니다.

상자를 그대로 SAM2.1에 넣으면 꽃받침과 잎까지 잡거나 비어 있었습니다. 그래서 상자 축소와 점 프롬프트로 후보를 여러 개 만들고, 가는 부분을 지운 뒤 가장 둥근 후보를 골랐습니다. 결과 47개는 Claude가 눈으로 확인해 모두 봉오리 몸통을 잡았습니다. 사용자는 아직 확인하지 않았습니다.

![첫 시도: 상자 프롬프트](runs/qa/masks_e1.jpg)

![개선 후: 봉오리 몸통](runs/qa/masks_body.jpg)

사진마다 달랐던 번호는 Claude가 생김새로 다시 맞췄습니다. F-6 사진은 A, B, C이고, 추가 사진은 P, R, G입니다. P는 A와 같아 보이지만, R과 G가 B, C 중 무엇인지는 판단하지 못했습니다.

## 학습

v3 A_base에서 시작해 ESP 사진만 학습했습니다. SGD, 학습률 0.002, 백본 10층 고정, 60에포크이고, 증강은 v3와 같습니다. 타일 880과 사진 전체 보기로 학습 타일 143장, 객체 78개, 검증 타일 26장, 객체 14개를 만들었습니다. 시드는 0, 1, 2입니다. 설정은 평가 전에 고정했고 다시 돌리지 않았습니다.

## 평가

찾음은 예측 상자의 중심이 사람이 그린 상자 안에 있는 경우이고, 봉오리 하나에 한 번만 셉니다. 다른 예측은 중심이 어떤 사람 상자에도 없는 예측입니다. IoU 0.5 열은 몸통 상자와 비교한 값입니다. v3 임계값은 기존 0.60이고, 파인튜닝 임계값은 검증 사진에서 찾음에서 다른 예측을 뺀 값이 가장 큰 임계값 중 가장 높은 값입니다.

| 조건 | 임계값 | 찾음 | 다른 예측 | IoU 0.5 | P | R | G |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows_t}

![평가 사진 비교](runs/report/test_compare.jpg)

임계값에 따른 찾은 횟수(다른 예측)입니다. 참고용이며 이 표로 임계값을 고르지 않습니다.

| 조건 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---:|---:|---:|---:|---:|---:|
{rows_s}

휴대폰 val 타일 결과입니다. 휴대폰 test는 열지 않았습니다.

| 모델 | 박스 mAP50 | 정밀도 | 재현율 | 마스크 mAP50 |
|---|---:|---:|---:|---:|
{rows_p}

## 한계

- **같은 봉오리입니다.** test의 P, R, G는 train에도 있습니다. 모델이 이 봉오리를 외웠을 수 있습니다.
- **표본이 작습니다.** test는 사진 3장, 봉오리 3개, 9번입니다.
- **검증이 쉬웠습니다.** val 2장은 train의 일부 사진과 배치가 비슷해 파인튜닝 모델이 모두 6/6이었습니다. 임계값 0.45~0.70이 동률이어서 규칙대로 가장 높은 값을 골랐고, 이 값이 test에서 찾은 횟수를 낮췄을 수 있습니다.
- **봉오리만 있습니다.** 개화중, 만개가 없어 큰 꽃과 단계 분류를 평가하지 못했습니다.
- **조명은 그대로입니다.** LED 상태를 바꾼 사진이 없습니다.
- **Jetson 시간을 재지 않았습니다.**
- **타일 880의 원인을 밝히지 못했습니다.** 왜 이번 평가에서 1280보다 낮은지 확인하지 않았습니다.
- **F-6 사진 2장은 가지도 움직인 것으로 보입니다.** 42e5f5, 817a7c에서 A와 C는 거의 그대로인데 B만 약 320px 움직였습니다. v3 리포트의 시점만 바꿨다는 설명은 이 점을 반영하지 않습니다.

## 다음 단계

- 다른 화분이나 새로 핀 봉오리 사진으로 평가합니다. 새 봉오리를 찾는지는 이 사진들로 알 수 없습니다.
- 꽃이 핀 상태와 LED 상태별 사진을 추가합니다.
- ESP 전용 가중치를 둘지, 휴대폰과 ESP 사진을 함께 학습할지 정합니다.
'''
(VER / 'REPORT.md').write_text(MD, encoding='utf-8')
CSS = 'body{font:15px/1.65 "Malgun Gothic","Noto Sans KR",sans-serif;background:#fcfcfb;color:#0b0b0b;margin:0}main{max-width:980px;margin:auto;padding:32px 20px}h1{font-size:28px}h2{font-size:20px;margin-top:36px;border-bottom:1px solid #e4e3df;padding-bottom:6px}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #e4e3df;padding:5px 8px}th{background:#f4f3ef}td.r,th.r{text-align:right}img{max-width:100%;border:1px solid #e4e3df;margin:8px 0}blockquote{display:none}'
inl = lambda s: re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html.escape(s, quote=False))
o = []; L = MD.split('\n'); i = 0
while i < len(L):
    l = L[i]
    if l.startswith('# '):
        o.append(f'<h1>{inl(l[2:])}</h1>')
    elif l.startswith('## '):
        o.append(f'<h2>{inl(l[3:])}</h2>')
    elif l.startswith('> '):
        pass
    elif l.startswith('!['):
        m = re.match(r'!\[(.*?)\]\((.*?)\)', l); o.append(f'<img alt="{html.escape(m[1])}" src="data:image/jpeg;base64,{base64.b64encode((VER / m[2]).read_bytes()).decode()}">')
    elif l.startswith('|'):
        rows = []
        while i < len(L) and L[i].startswith('|'):
            rows.append([c.strip() for c in L[i].strip('|').split('|')]); i += 1
        al = ['r' if c.endswith(':') else '' for c in rows[1]]; o.append('<table><tr>' + ''.join(f'<th class="{a}">{inl(c)}</th>' for a, c in zip(al, rows[0])) + '</tr>' + ''.join('<tr>' + ''.join(f'<td class="{a}">{inl(c)}</td>' for a, c in zip(al, r)) + '</tr>' for r in rows[2:]) + '</table>'); continue
    elif l.startswith('- '):
        it = []
        while i < len(L) and L[i].startswith('- '):
            it.append(f'<li>{inl(L[i][2:])}</li>'); i += 1
        o.append('<ul>' + ''.join(it) + '</ul>'); continue
    elif l.strip():
        o.append(f'<p>{inl(l)}</p>')
    i += 1
(VER / 'REPORT.html').write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>꽃 모델 v4 리포트</title><style>{CSS}</style></head><body><main>' + ''.join(o) + '</main></body></html>', encoding='utf-8'); print('REPORT_DONE', flush=True)
