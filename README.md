# 미니장미 꽃 모델

스마트큐브와 농장의 장미 사진에서 꽃을 찾아 꽃마다 번호를 매기고, 봉오리, 개화중, 만개를 판정하는 모델입니다. 목적은 큐브 안의 꽃 상태 확인입니다. 잎 모델과 다른 모델이라 저장소를 따로 둡니다. 잎 모델은 [Lumiela/rose_leaf](https://github.com/Lumiela/rose_leaf)에 있습니다.

- **이 저장소는 private입니다.** 저장소 소유자에게 접근 권한을 받아야 clone할 수 있습니다. 농장 사진이 들어 있으니 팀 밖으로 공유하지 않습니다.
- 버전마다 폴더가 하나입니다. 각 버전 폴더의 `REPORT.html`은 그림이 들어간 리포트이고 `REPORT.md`는 요약본입니다. HTML은 내려받아 브라우저로 엽니다.

## 현재 상태 (2026-11-06)
| 항목 | 내용 |
|---|---|
| 기본 모델 (휴대폰 사진) | 꽃 찾기 `weights/flower_seg_v3.pt` (v3_recall A_base) + 단계 분류 `weights/flower_cls_v1r2.pt` (v1_box effnetv2s_r2) |
| ESP 카메라용 | 꽃 찾기 `weights/flower_seg_v4esp.pt` (v4_esp ft_s2) + 같은 단계 분류기, 타일 880, 임계값 0.70 |
| 휴대폰 test (17장, 라벨 550개) | 정밀도 0.905, 재현율 0.913, F1 0.909, 사진당 봉오리 개수 오차 2.65, 단계까지 본 매크로 F1 0.760 ([v3 리포트](v3_recall/REPORT.md)) |
| ESP test (3장, 봉오리 9번) | v4 ft_s2가 5번 찾음, 다른 예측 0개. v3를 그대로 쓰면(타일 1280) 2번 ([v4 리포트](v4_esp/REPORT.md)) |
| 최종 모델 | **미정.** 위 가중치는 지금 쓰는 기본값입니다 |
| 다음 작업 | 시드 반복 학습, v3 오탐 53개 사람 판정, 단계 분류기 개선 |

## 빠른 실행
### 1. 받기
```bash
git clone git@github.com:Lumiela/rose_detect.git
cd rose_detect
```
HTTPS 주소는 `https://github.com/Lumiela/rose_detect.git`입니다. 사진 650MB와 가중치 130MB가 들어 있어 clone하면 약 1.7GB입니다.

### 2. 설치
Python 3.10에서 확인했습니다.
```bash
pip install -r requirements.txt
```
NVIDIA GPU를 쓰려면 [pytorch.org](https://pytorch.org/get-started/locally/) 안내대로 CUDA용 torch, torchvision을 먼저 설치한 뒤 위 명령을 실행합니다. GPU가 없으면 자동으로 CPU에서 돌고, 결과는 같습니다.

### 3. 실행
```bash
# 휴대폰 사진 (사진 한 장 또는 폴더)
python v3_recall/scripts/40_flower_ids.py <사진 또는 폴더> <출력 폴더>

# ESP 카메라 사진
python v3_recall/scripts/40_flower_ids.py <사진 또는 폴더> <출력 폴더> --camera esp --seg weights/flower_seg_v4esp.pt --conf 0.70
```
- **`--sheet`:** 꽃마다 잘라 붙인 확인용 시트를 함께 저장합니다.
- **`--conf`:** 꽃으로 인정하는 최소 신뢰도입니다. 기본값은 `cameras.json`의 값입니다.
- **읽는 사진:** `.jpg`, `.jpeg`, `.jfif`, `.png`. 폴더를 주면 하위 폴더는 읽지 않습니다.
- 위 명령은 저장소 최상위 폴더에서 실행하는 기준입니다. 가중치 경로를 절대 경로로 주면 다른 폴더에서도 됩니다.

### 4. 설치 확인
저장소에 든 사진으로 아래와 같이 나오면 정상입니다.
```bash
python v3_recall/scripts/40_flower_ids.py common/photos/20261030_134905.jpg out_check
python v3_recall/scripts/40_flower_ids.py v4_esp/photos/8583c9e8-3806-4e5c-a67f-dc169a3b303c.jfif out_check_esp --camera esp --seg weights/flower_seg_v4esp.pt --conf 0.70
```
| 사진 | 마지막 줄 | 결과 |
|---|---|---|
| 휴대폰 `20261030_134905.jpg` (3000x4000) | `IDS_DONE 1 41` | 41개: 봉오리 19, 개화중 12, 만개 10 |
| ESP `8583c9e8...jfif` (2560x1920) | `IDS_DONE 1 1` | 1개: 봉오리 1 |

휴대폰 사진 한 장에 서버 V100으로 탐지 3.8초, 분류 2.5초가 걸렸습니다. GPU를 끄고 서버 CPU로 돌리면 탐지 5.9초, 분류 2.9초였고, 일반 PC의 CPU는 이보다 느릴 수 있습니다. GPU와 CPU 결과는 개수와 단계가 같고, 단계 확률만 0.004 이내로 다릅니다.

## 출력
| 파일 | 내용 |
|---|---|
| `<사진>_ids.jpg` | 꽃마다 마스크와 번호를 그린 그림. 위쪽 띠에 단계별 개수. 초록 봉오리(bud), 노랑 개화중(opening), 분홍 만개(full) |
| `<사진>_sheet.jpg` | `--sheet`를 줬을 때. 꽃마다 크롭, 번호, 단계, 확률 |
| `flowers.csv` | 꽃 하나가 한 줄 |
| `flowers.json` | `flowers.csv`의 내용에 마스크 다각형(`poly`)을 더한 것 |
| `summary.csv` | 사진별 개수와 걸린 시간 |

| `flowers.csv` 열 | 뜻 |
|---|---|
| `id`, `idx` | `<사진 이름>_F0001` 형식의 번호. 사진 안에서 위에서 아래, 왼쪽에서 오른쪽 순서입니다. **다른 사진의 같은 꽃과 이어지지 않습니다** (추적 없음) |
| `x0`, `y0`, `x1`, `y1`, `cx`, `cy` | 박스와 중심. 원본 사진의 픽셀 좌표이고 EXIF 회전을 적용한 기준입니다 |
| `conf` | 꽃 탐지 신뢰도 |
| `stage`, `stage_idx` | 봉오리, 개화중, 만개 / 0, 1, 2 |
| `stage_prob`, `prob_bud`, `prob_opening`, `prob_full` | 단계 확률 |
| `area`, `solidity` | 마스크 넓이(px)와 볼록도. 볼록도 0.8 미만은 대부분 잎에 가려진 꽃입니다 |
| `touches_border` | 사진 가장자리에 닿은 꽃 |

## 카메라별 설정
모델 입력은 640으로 같고, 사진을 자르는 타일 크기만 다릅니다. 봉오리가 입력에서 학습 때와 비슷한 크기가 되게 하려는 것입니다. 설정은 [`v3_recall/cameras.json`](v3_recall/cameras.json)에 있습니다.

| `--camera` | 사진 | 타일 | 기본 임계값 | 근거 |
|---|---|---:|---:|---|
| `phone` (기본) | 휴대폰 3000x4000 | 1280 | 0.60 | 학습과 평가의 기준 |
| `esp` | ESP 카메라 2560x1920 | 880 | 0.60 | v3 가중치용. 사진 6장, 봉오리 3개로 정한 임시 값 |

ESP 가중치(`flower_seg_v4esp.pt`)를 쓸 때는 val에서 정한 `--conf 0.70`을 줍니다.

## 결과를 읽을 때 주의
- **휴대폰 test:** 하루, 한 농장에서 휴대폰으로 찍은 사진입니다. 화분 단위로 나누지 않아서 실제보다 높게 나올 수 있습니다.
- **ESP test:** 사진 3장에서 봉오리 3개를 배치만 바꿔 쟀습니다. test 봉오리가 train에도 있어서, 배치가 바뀌어도 찾는지만 보여 줍니다.
- **ESP 가중치는 ESP 전용입니다.** 휴대폰 val의 box mAP50이 0.96에서 0.85로 떨어집니다.
- **ESP 가중치 시드:** ft_s0~2가 ESP val에서 모두 6번 중 6번을 찾아 동률이라, 휴대폰 val mAP50이 가장 높은 ft_s2를 골랐습니다. test 결과(5, 6, 5번)는 고르는 데 쓰지 않았습니다.
- **단계 분류기:** 휴대폰 사진으로만 학습했습니다. ESP 사진의 단계 판정은 검증 전입니다.

## 가중치
| 파일 | 원본 | 용도 | 크기 | md5 |
|---|---|---|---:|---|
| `weights/flower_seg_v3.pt` | `v3_recall/runs/seg/A_base/weights/best.pt` | 꽃 찾기 (YOLO26s-seg, 1클래스) | 23 MB | `22b81367d22f45816b0438d82f8b5ffb` |
| `weights/flower_cls_v1r2.pt` | `v1_box/runs/cls/effnetv2s_r2/best.pt` | 단계 분류 (EfficientNetV2-S, 박스 긴 변 x 1.25 크롭을 224로 줄임) | 82 MB | `5aa7dffc623fc330bd6cdd1a7e2effac` |
| `weights/flower_seg_v4esp.pt` | `v4_esp/runs/seg/ft_s2/weights/best.pt` | ESP 사진 꽃 찾기 | 23 MB | `71201008872b0b0245efa3611d74c5cb` |

- **git에 없는 것:** v1, v2 가중치, 학습 중간 결과(`runs/`), 내보낸 학습 데이터(`data/`)는 넣지 않았습니다. 스크립트와 라벨로 다시 만듭니다.
- **사전학습 가중치:** 학습할 때만 필요하고 git에 없습니다. 서버의 `common/weights/`에 `yolo26s-seg.pt`, `yolo26n.pt`, `sam2.1_s.pt`, torchvision의 `efficientnet_v2_s` ImageNet 가중치가 있었고, 모두 Ultralytics와 torchvision에서 다시 받을 수 있습니다.

## 버전
| 버전 | 구조 | 상태 |
|---|---|---|
| [v1_box](v1_box/README.md) ([리포트](v1_box/REPORT.md)) | 박스 탐지 + 박스별 단계 분류 | 후보 `det/third_pseudo`, `cls/effnetv2s_r2`. 이 분류기를 지금도 씁니다 |
| [v2_seg](v2_seg/README.md) ([리포트](v2_seg/REPORT.md)) | 세그멘테이션 + ID + ID별 단계 분류 | 학습, ID 부여, [v1 비교](v2_seg/COMPARE.md) 완료 |
| [v3_recall](v3_recall/README.md) ([리포트](v3_recall/REPORT.md)) | v2 구조, 팀 교차 판정 라벨로 재학습 | A_base 학습과 test 평가 완료. 지금 기본 꽃 찾기 모델 |
| [v4_esp](v4_esp/README.md) ([리포트](v4_esp/REPORT.md)) | v3를 ESP 사진으로 파인튜닝 | 평가 완료. 같은 봉오리로 잰 결과라 사용 여부 미정 |

v3까지의 점수는 휴대폰 사진 기준입니다. 사람이 라벨을 그린 ESP32 사진 평가는 v4_esp에만 있습니다.

## 폴더 지도
| 경로 | 내용 |
|---|---|
| `weights/` | 추론 가중치 3개 |
| `requirements.txt` | 실행에 필요한 패키지 |
| `common/photos/` | 농장 휴대폰 사진 143장. EXIF 위치 정보를 지운 사본입니다 |
| `common/paths.py` | opencv 우회와 사전학습 가중치 경로. 모든 스크립트가 먼저 불러옵니다 |
| `v1_box/` | 라벨 원본(고정), 분할 기준, 라벨 규칙, 단계 분류기 학습 |
| `v2_seg/` | SAM2.1 마스크, 세그멘테이션 학습, ID 부여, v1 비교 |
| `v3_recall/` | 최신 라벨 `labels/`, 추론 `scripts/40_flower_ids.py`와 `infer.py`, `cameras.json` |
| `v4_esp/` | ESP 사진 17장 `photos/`, 사람 라벨, 파인튜닝과 평가 스크립트 |
| `human_label/` | 사람 판정 회차 기록. 판정 HTML은 농장 사진 조각이 들어 있어 git에 넣지 않았습니다 |

## 데이터
- **휴대폰 사진:** 2026-10-30, 한 농장, 143장. 휴대폰 두 대가 같은 시각에 다른 재배대를 찍어 시각 구간 단위로 나눴습니다. train 108, val 8, test 18(이 중 온실 전경 1장은 평가에서 뺌), buffer 9(어디에도 쓰지 않음). 자세한 기준은 [v1_box/README.md](v1_box/README.md)에 있습니다.
- **라벨:** 최신 라벨은 `v3_recall/labels/annotations.csv`입니다. 라벨별 출처와 사람 확인 범위는 같은 폴더의 `provenance.csv`에 있습니다.
- **라벨 규칙:** 작은 봉오리, 초록 봉오리, 흰 꽃까지 모두 표시하고 박스는 꽃받침까지 포함합니다. 단계 기준은 [v1_box/README.md](v1_box/README.md)를 봅니다.
- **ESP 사진:** 2560x1920, 17장, 사람 라벨 47개(모두 봉오리). train 11, val 2, test 3, 제외 1장입니다.

## 버전 관리 규칙
- **`vN_이름/`:** 버전마다 한 폴더입니다. 이전 버전은 고치지 않습니다. 라벨은 최신 버전 폴더에 두고 이전 버전의 라벨은 고정합니다. `v1_box/labels/`는 v1, v2가 쓰는 원본이고, 최신 라벨은 `v3_recall/labels/`입니다.
- **가중치:** 추론에 쓰는 것만 `weights/`에 넣습니다. 바꿀 때는 버전이 들어간 새 이름으로 추가하고 이 README의 표를 고칩니다.
- **리포트:** 중요 작업이 끝나면 작성 여부를 먼저 묻고, 버전 폴더의 `REPORT.html`과 요약 `REPORT.md`에 씁니다.
- **태그:** 잎 모델의 `v1`~`v10`과 겹치지 않게 `flower-vN`으로 답니다. 아직 단 태그는 없습니다.
- **`human_label/`:** 회차마다 `html/`과 `labels/`를 둡니다.

## 주의
- **opencv 충돌:** 서버에는 opencv 4.7과 4.11이 섞여 있어, `ultralytics`를 불러오기 전에 `common/paths.py`를 먼저 불러와야 합니다. 스크립트는 이미 그렇게 되어 있습니다. 직접 코드를 짤 때도 `import paths`를 먼저 합니다.
- **옛 경로:** `runs/*/args.yaml`과 커밋 메시지에 `model_git/v11_flower` 같은 당시 경로가 남아 있습니다. 기록용입니다.
- **학습 재현:** 학습 스크립트는 `data/`를 다시 내보낸 뒤 실행합니다. GPU 서버를 기준으로 작성했습니다.
- **확인 범위:** 위 실행은 서버(Linux, V100)에서 GPU와 CPU로, 새로 clone한 저장소로 확인했습니다. 새 PC에 처음 설치하는 과정과 Windows 실행은 확인하지 않았습니다. 확인한 버전은 `requirements.txt`에 적었습니다.
- **push:** 서버에는 GitHub 인증이 없어, 올릴 때마다 임시 deploy key를 만들고 끝나면 지웁니다. 한 번에 2GB를 넘게 올릴 수 없습니다.
