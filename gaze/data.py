"""데이터 읽기와 fold 분할.

이미지는 reader.loader로 읽고(0~1 float32), 반복 실행을 빠르게 하려고 uint8 npz로 캐시한다.
표준화 통계는 fold마다 학습용 사람들로만 계산한다 (README의 평가 규칙).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import reader

PERSONS = [f"p{k:02d}" for k in range(15)]
DATA_ROOT = "data/EyeBased/MPIIGaze"
CACHE_PATH = "data/mpiigaze_eye.npz"


def read_label_table(root, person):
    """라벨 파일의 모든 칸을 {헤더: 문자열 배열} 사전으로 돌려준다 (EDA용)."""
    with open(os.path.join(root, "Label", f"{person}.label")) as f:
        header = f.readline().split()
        rows = [ln.split() for ln in f if ln.strip()]
    cols = list(zip(*rows))
    return {name: np.array(col) for name, col in zip(header, cols)}


def load_person(root, person, num_workers=8):
    """한 사람의 (eye uint8 (N,36,60), head (N,2), gaze (N,2))."""
    dataset = reader.loader(os.path.join(root, "Label", f"{person}.label"), os.path.join(root, "Image"))
    with ThreadPoolExecutor(num_workers) as pool:
        samples = list(pool.map(dataset.__getitem__, range(len(dataset))))
    info, gaze = reader.collate(samples)
    eye = np.rint(info["eye"][:, 0] * 255).astype(np.uint8)  # reader가 /255 한 값을 그대로 되돌림
    return eye, info["head_pose"].astype(np.float32), gaze.astype(np.float32)


def load_all(root=DATA_ROOT, cache=CACHE_PATH, persons=PERSONS, verbose=True):
    """{person: (eye uint8, head, gaze)}. cache가 있으면 거기서 읽고, 없으면 만든다."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        data = {p: (z[f"{p}_eye"], z[f"{p}_head"], z[f"{p}_gaze"]) for p in persons}
        if verbose:
            print(f"loaded cache {cache}: {sum(len(d[0]) for d in data.values())} samples")
        return data

    data = {}
    for p in persons:
        data[p] = load_person(root, p)
        if verbose:
            print(f"  {p}: {len(data[p][0])} samples")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        arrays = {}
        for p, (eye, head, gaze) in data.items():
            arrays[f"{p}_eye"], arrays[f"{p}_head"], arrays[f"{p}_gaze"] = eye, head, gaze
        np.savez(cache, **arrays)
        if verbose:
            print(f"saved cache {cache}")
    return data


def stack(data, persons):
    """여러 사람을 이어 붙인 (eye, head, gaze)."""
    eye = np.concatenate([data[p][0] for p in persons])
    head = np.concatenate([data[p][1] for p in persons])
    gaze = np.concatenate([data[p][2] for p in persons])
    return eye, head, gaze


def split_fold(test, persons=PERSONS, use_val=True):
    """leave-one-person-out 분할. 검증용 사람은 학습용 14명 중 test 다음 번호 한 명."""
    rest = [p for p in persons if p != test]
    if not use_val:
        return rest, [], test
    val = persons[(persons.index(test) + 1) % len(persons)]
    train = [p for p in rest if p != val]
    return train, [val], test


class Standardizer:
    """학습 데이터로만 평균·표준편차를 구해 입력과 정답을 표준화한다.

    - 이미지: 픽셀 전체의 평균·표준편차 (스칼라)
    - head pose, gaze: (yaw, pitch) 각각
    """

    def fit(self, eye, head, gaze):
        x = eye.astype(np.float32) / 255.0
        self.eye_mean, self.eye_std = float(x.mean()), float(x.std())
        self.head_mean, self.head_std = head.mean(0), head.std(0)
        self.gaze_mean, self.gaze_std = gaze.mean(0), gaze.std(0)
        return self

    def x(self, eye, head):
        """모델 입력: (N,1,36,60), (N,2) float32."""
        img = (eye.astype(np.float32) / 255.0 - self.eye_mean) / self.eye_std
        h = (head - self.head_mean) / self.head_std
        return img[:, None].astype(np.float32), h.astype(np.float32)

    def t(self, gaze):
        return ((gaze - self.gaze_mean) / self.gaze_std).astype(np.float32)

    def inverse_t(self, t):
        return t * self.gaze_std + self.gaze_mean

    def state(self):
        return {k: np.asarray(v).tolist() for k, v in vars(self).items()}


def raw_frame_counts(dataset_root="data/MPIIGaze"):
    """원본 Data/Original의 사람·day별 프레임 수 {person: {day: n}} (annotation.txt 줄 수)."""
    orig = os.path.join(dataset_root, "Data", "Original")
    counts = {}
    for p in sorted(os.listdir(orig)):
        pdir = os.path.join(orig, p)
        if not p.startswith("p") or not os.path.isdir(pdir):
            continue
        counts[p] = {}
        for day in sorted(os.listdir(pdir)):
            txt = os.path.join(pdir, day, "annotation.txt")
            if day.startswith("day") and os.path.isfile(txt):
                with open(txt) as f:
                    counts[p][day] = sum(1 for ln in f if ln.strip())
    return counts
