# MPIIGaze 시선 추정: EDA와 교재 코드 기반 파이프라인

[ssuai/gaze_estimation](https://github.com/ssuai/gaze_estimation)을 fork해 다음을 추가함.

1. **EDA** — [eda.ipynb](eda.ipynb)
2. **파이프라인** — 『밑바닥부터 시작하는 딥러닝』 교재 코드(`common/`, numpy)만으로 만든 CNN을 leave-one-person-out으로 학습·평가. [pipeline.ipynb](pipeline.ipynb), [train_lopo.py](train_lopo.py)

두 노트북은 실제 MPIIGaze 데이터로 실행한 결과가 저장되어 있음 (GitHub Actions에서 다운로드·전처리·학습·실행).

## 결과: leave-one-person-out 각도 오차 (°)

| 방법 | p00 | p01 | p02 | p03 | p04 | p05 | p06 | p07 | p08 | p09 | p10 | p11 | p12 | p13 | p14 | 평균 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| 평균 gaze 예측 | 8.42 | 8.61 | 9.00 | 9.60 | 8.97 | 9.73 | 10.36 | 9.50 | 9.62 | 8.97 | 8.66 | 8.43 | 8.60 | 9.00 | 10.76 | 9.22 |
| ridge 회귀 (수업 baseline) | 5.31 | 7.76 | 6.55 | 8.05 | 6.40 | 7.35 | 7.59 | 7.97 | 7.89 | 8.51 | 6.93 | 6.63 | 7.07 | 7.85 | 7.20 | 7.27 |
| **GazeLeNet (교재 numpy CNN)** | 5.09 | 6.73 | 6.52 | 6.11 | 6.66 | 5.74 | 6.53 | 6.99 | 6.52 | 7.81 | 7.26 | 5.70 | 5.68 | 6.95 | 5.75 | **6.40** |

- 15명 중 13명에서 ridge보다 낮음 (p04, p10은 조금 높음)
- 평균 gaze 예측은 fold마다 학습 13명의 평균을 씀 (아래 검증 분할 때문). 14명 평균으로 계산한 값은 9.21°로 README 아래 표와 같음
- 원본 수치: [results/lopo_results.csv](results/lopo_results.csv), fold별 학습 기록: [results/lopo/](results/lopo/)

## 파이프라인

```
eye (1,36,60) → Conv(20,5×5) → ReLU → Pool → Conv(50,5×5) → ReLU → Pool → Affine(500) → ReLU ─┐
head pose (2) ─────────────────────────────────────────────────────────────────────────────────┴→ Concat → Affine(2) → MSE
```

- 구조는 MPIIGaze 논문(Zhang et al., CVPR 2015)의 CNN과 같음
- 계층·im2col·옵티마이저는 교재 `common/`을 그대로 씀. 바꾼 곳은 float32 유지를 위한 dtype 3줄뿐 ([common/\_\_init\_\_.py](common/__init__.py))
- 새로 만든 계층: `Concat`(head pose 결합), `MeanSquaredLoss`(회귀 손실) — [gaze/layers.py](gaze/layers.py). 수치 미분으로 기울기 확인 (pipeline.ipynb 2.2절)
- 학습: Adam(lr 5e-4, 에폭마다 ×0.8), 배치 128, 10 에폭, He 초깃값
- 평가 규칙(README 아래 "과제" 절)을 지킴
  - 테스트하는 사람은 학습·표준화·에폭 선택에 쓰지 않음
  - 검증은 학습용 14명 중 한 명(테스트 다음 번호)을 사람 단위로 떼어 내 에폭 선택에만 씀
  - 하이퍼파라미터는 모든 fold에 같음 ([gaze/experiment.py](gaze/experiment.py) `CONFIG`), fold마다 새로 초기화

| 파일 | 내용 |
|------|------|
| `common/` | 교재 공통 모듈 (MIT, Koki Saitoh) |
| `gaze/data.py` | 라벨·이미지 읽기, npz 캐시, fold 분할, 표준화 |
| `gaze/model.py` | `GazeLeNet` (교재 `SimpleConvNet` 확장) |
| `gaze/trainer.py` | 미니배치 학습 루프, 검증 기반 에폭 선택 |
| `gaze/experiment.py` | fold 하나 실행 (`run_fold`) |
| `train_lopo.py` | 15-fold 학습·요약 CLI |
| `.github/workflows/pipeline.yml` | 다운로드·전처리 → fold별 병렬 학습 → 노트북 실행 → 결과 커밋 |

### 실행

```
pip install numpy opencv-python scipy matplotlib pandas jupyter
python -m preprocess.mpii
python train_lopo.py
python train_lopo.py --summarize
```

- CPU numpy라 fold 하나에 20분 정도 걸림. `python train_lopo.py --folds p00`처럼 일부만 돌릴 수 있음
- 노트북은 Colab 배지로도 열 수 있음 (첫 셀이 저장소 clone과 데이터 전처리를 함)

## EDA 요약

- 45,000개(15명 × 왼눈·오른눈 1,500개씩), 결측 없음. 원본 213,658 프레임 중 일부만 쓰며 사람마다 기록 기간(7~67일)이 크게 다름
- histogram equalization으로 밝기 분포는 고르지만, 눈 모양·안경·흐림 같은 외형 차이가 사람마다 뚜렷함 → 사람 단위 일반화가 핵심
- gaze는 yaw 0.0° ± 8.7°, pitch −8.8° ± 5.2°로 범위·중심이 달라 정답을 축별로 표준화함
- gaze와 head pose의 상관이 작음(|r| ≤ 0.11). 이미지가 주된 정보원이고, 픽셀–gaze 상관은 홍채 양옆(yaw)과 눈꺼풀(pitch)에 모여 있음 → CNN이 적합
- **오른눈 head yaw가 0°로 몰림** (표준편차 왼눈 11.6° vs 오른눈 1.6°). 전처리 `head_flip`이 회전 행렬 일부만 뒤집어 yaw 정보가 사라지는 것으로 보임 (eda.ipynb 6.1절에서 재현). 수업 baseline과 같은 데이터로 비교하려고 전처리는 고치지 않음

---

## 데이터 폴더 구조

`python -m preprocess.mpii`를 실행하면 `data/` 아래에 다음이 생긴다. `data/`는 `.gitignore`에 들어 있어 저장소에 올라가지 않는다. 변환 과정은 [preprocess/README.md](preprocess/README.md)에 정리되어 있다.

```
data/
├── raw/
│   └── MPIIGaze.tar.gz                  # 원본 아카이브 (2.0G). 있으면 다시 받지 않는다
├── MPIIGaze/                            # 압축 해제한 원본 (5.2G)
│   ├── ReadMe.txt
│   ├── 6 points-based face model.mat
│   ├── Annotation Subset/p00.txt … p14.txt
│   ├── Data/
│   │   ├── Original/p00 … p14/          # 전처리 입력 (4.4G)
│   │   │   ├── Calibration/             # Camera.mat, monitorPose.mat, screenSize.mat
│   │   │   └── day01 … dayNN/           # 0001.jpg … (웹캠 프레임) + annotation.txt
│   │   └── Normalized/p00 … p14/        # day01.mat … (데이터셋 제공 정규화본, 쓰지 않음)
│   └── Evaluation Subset/
│       ├── ReadMe.txt
│       ├── annotation for face image/p00.txt … p14.txt
│       └── sample list for eye image/p00.txt … p14.txt   # 변환할 샘플 목록
└── EyeBased/
    └── MPIIGaze/                        # 전처리 결과 (190M), 학습에 쓰는 데이터
        ├── Image/p00 … p14/1.jpg … 3000.jpg   # 36×60 grayscale 눈 이미지
        └── Label/p00.label … p14.label        # 샘플당 한 줄, 첫 줄은 헤더
```

- 전처리가 읽는 원본은 `Data/Original`의 `Calibration/Camera.mat`(카메라 내부 행렬), 각 day의 `annotation.txt`와 프레임, 그리고 `Evaluation Subset/sample list for eye image`의 목록뿐이다. 나머지는 데이터셋에 함께 들어 있는 파일이다.
- day 폴더 수는 사람마다 다르다 (7~69개).
- 결과는 사람당 3,000개(왼눈 1,500 + 오른눈 1,500), 15명 합계 45,000개다. 오른눈 이미지는 좌우 반전되어 왼눈처럼 저장된다.
- 이미지 파일 이름은 목록 순서대로 붙인 일련번호이며, 원본 프레임 위치는 라벨의 `Origin` 칸에 남는다.

라벨 예 (`Label/p00.label`, 소수는 줄임):

```
Image Origin WhichEye 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin
p00/1.jpg day08/0069.jpg left -0.507,-0.112,-0.029 0.247,-0.052 -0.115,0.508 -0.023,-0.171,-0.037 1.0,1.0,1.205 0.0,0.0,600.0
```

`reader.py`는 `loader("data/EyeBased/MPIIGaze/Label/p00.label", "data/EyeBased/MPIIGaze/Image")`처럼 라벨 파일과 `Image/` 폴더를 받는다. 각 칸의 의미는 [preprocess/README.md의 5. 출력](preprocess/README.md#5-출력)을 본다.


## 과제

눈 이미지와 head pose가 주어졌을 때 시선 방향 `2DGaze`를 추정한다.

### 입력과 출력

| 구분 | 라벨 칸 | 형태 | 내용 |
|------|---------|------|------|
| 입력 | `Image` | (1, 36, 60) float32, 0~1 | 정규화된 grayscale 눈 이미지 |
| 입력 | `2DHead` | (2,) float32 | head pose (yaw, pitch), 라디안 |
| 출력 | `2DGaze` | (2,) float32 | gaze (yaw, pitch), 라디안 |

- 모델은 이미지와 head pose **둘 다** 입력으로 받는다.
- 라벨의 나머지 칸(`Origin`, `WhichEye`, `3DHead`, `Rmat`, `Smat`, `GazeOrigin`)은 입력으로 쓰지 않는다.
- 왼눈과 오른눈은 서로 다른 샘플로 다룬다. 오른눈은 전처리에서 이미 좌우 반전되어 있으므로 두 눈을 같은 모델로 학습한다.

`reader.py`가 이 형태 그대로 돌려준다.

```python
info, label = dataset[i]
info["eye"]        # 입력 이미지
info["head_pose"]  # 입력 head pose
label              # 정답 2DGaze
```

### 평가: leave-one-person-out

15명(p00~p14)을 한 명씩 돌아가며 테스트 대상으로 삼는 15-fold 교차 검증이다.

1. fold k에서는 `pk`의 3,000개를 테스트 세트로, 나머지 14명의 42,000개를 학습 세트로 쓴다.
2. 모델을 학습 세트로 처음부터 학습하고, 테스트 세트에서 평균 각도 오차를 잰다.
3. 15개 fold의 결과를 모두 보고하고, 그 평균을 최종 성능으로 삼는다.

```python
persons = [f"p{k:02d}" for k in range(15)]
label = lambda p: f"data/EyeBased/MPIIGaze/Label/{p}.label"

for test in persons:
    train = reader.txtload([label(p) for p in persons if p != test], "data/EyeBased/MPIIGaze/Image", batch_size=..., shuffle=True)
    test_loader = reader.txtload(label(test), "data/EyeBased/MPIIGaze/Image", batch_size=..., shuffle=False)
    ...
```

주의할 점:

- 테스트하는 사람의 데이터는 학습, 정규화 통계 계산, early stopping, 하이퍼파라미터 선택 어디에도 쓰지 않는다. 검증 세트가 필요하면 학습용 14명 안에서 사람 단위로 떼어 낸다.
- 하이퍼파라미터는 모든 fold에서 같게 둔다. fold마다 따로 고르지 않는다.
- fold마다 모델을 새로 초기화한다.

### 평가 지표: 각도 오차

예측과 정답 (yaw, pitch)를 각각 3D 단위 벡터로 바꾼 뒤 두 벡터 사이의 각도를 도(°) 단위로 잰다. yaw와 pitch의 오차를 따로 재거나 2D 거리로 재지 않는다.

```python
import numpy as np
from preprocess.mpii_norm import gaze_to_3d

def angular_error(pred, gt):
    """pred, gt: (N, 2) (yaw, pitch) 라디안 → (N,) 도"""
    a, b = gaze_to_3d(pred), gaze_to_3d(gt)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
```

결과는 다음처럼 보고한다.

| 테스트 대상 | p00 | p01 | … | p14 | 평균 |
|-------------|-----|-----|---|-----|------|
| 각도 오차 (°) | | | | | |

### Baseline: 선형 회귀

[baseline_linear.py](baseline_linear.py)는 numpy만으로 만든 ridge 회귀다. 눈 이미지 픽셀 2,160개와 `2DHead` 2개를 표준화해 이어 붙이고, 정규 방정식을 풀어 `2DGaze`를 바로 예측한다. 평균·표준편차와 회귀 계수는 fold마다 학습용 14명으로만 계산한다. 전체 15 fold가 CPU에서 20초 안에 끝난다.

```bash
python baseline_linear.py                # alpha=1000, 이미지 + 2DHead
python baseline_linear.py --no-head      # 이미지만
```

leave-one-person-out 각도 오차 (°):

| 방법 | p00 | p01 | p02 | p03 | p04 | p05 | p06 | p07 | p08 | p09 | p10 | p11 | p12 | p13 | p14 | 평균 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| 평균 gaze 예측 | 8.43 | 8.63 | 9.00 | 9.60 | 8.97 | 9.73 | 10.34 | 9.49 | 9.61 | 8.97 | 8.66 | 8.43 | 8.60 | 9.00 | 10.75 | 9.21 |
| ridge 회귀 (alpha=1000) | 5.31 | 7.76 | 6.55 | 8.05 | 6.40 | 7.35 | 7.59 | 7.97 | 7.89 | 8.51 | 6.93 | 6.63 | 7.07 | 7.85 | 7.20 | **7.27** |

- 평균 gaze 예측은 입력을 보지 않고 학습용 14명의 `2DGaze` 평균을 항상 내놓는 하한선이다.
- `2DHead`를 빼면(`--no-head`) 평균 7.28°로, 선형 모델에서는 head pose가 거의 도움이 되지 않는다.
- alpha에 따른 평균 오차: 10 → 7.42°, 100 → 7.37°, 1000 → 7.27°, 10000 → 7.16°, 100000 → 7.05°. 이 값은 테스트 결과로 비교한 것이므로 여기서 alpha를 고르면 테스트 데이터를 보고 하이퍼파라미터를 정하는 셈이 된다. 기본값은 1000으로 둔다.


## Installation (conda)

```
conda create -n gaze python=3.12 -y
conda activate gaze

conda install -c conda-forge jupyter matplotlib # for visualization and Eda
```
