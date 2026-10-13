"""leave-one-person-out 한 fold를 처음부터 끝까지 돌리는 함수. CLI(train_lopo.py)와 노트북이 함께 쓴다."""
import json
import os

import numpy as np

from gaze.data import PERSONS, Standardizer, split_fold, stack
from gaze.metrics import angular_error
from gaze.model import GazeLeNet
from gaze.trainer import evaluate, train

# 모든 fold에 같은 하이퍼파라미터를 쓴다 (README 규칙). 테스트 결과를 보고 고르지 않는다.
CONFIG = {
    "conv_params": [[20, 5], [50, 5]],
    "hidden_size": 500,
    "dropout_ratio": 0.0,
    "use_head": True,
    "epochs": 10,
    "batch_size": 128,
    "optimizer": "adam",
    "lr": 5e-4,
    "lr_decay": 0.8,
    "seed": 0,
    "use_val": True,
}


def make_set(std, eye, head, gaze, use_head=True):
    x, h = std.x(eye, head)
    if not use_head:
        h = np.zeros_like(h)  # head pose를 빼는 비교 실험: 입력 자리를 0으로 둔다
    return {"x": x, "head": h, "t": std.t(gaze), "gaze": gaze}


def run_fold(data, test, config=None, verbose=True, return_model=False):
    cfg = {**CONFIG, **(config or {})}
    train_p, val_p, test_p = split_fold(test, PERSONS, use_val=cfg["use_val"])
    tr = stack(data, train_p)
    std = Standardizer().fit(*tr)  # 학습용 사람들로만
    train_set = make_set(std, *tr, use_head=cfg["use_head"])
    val_set = make_set(std, *stack(data, val_p), use_head=cfg["use_head"]) if val_p else None
    test_set = make_set(std, *data[test_p], use_head=cfg["use_head"])

    if verbose:
        print(f"[fold {test}] train {train_p[0]}..({len(train_p)}명, {len(tr[0])}개)  "
              f"val {val_p}  test {test_p}", flush=True)
    net = GazeLeNet(conv_params=cfg["conv_params"], hidden_size=cfg["hidden_size"],
                    dropout_ratio=cfg["dropout_ratio"], seed=cfg["seed"])
    hist = train(net, std, train_set, val_set, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                 optimizer=cfg["optimizer"], lr=cfg["lr"], lr_decay=cfg["lr_decay"], seed=cfg["seed"],
                 verbose=verbose)

    test_err, pred = evaluate(net, std, test_set["x"], test_set["head"], test_set["gaze"])
    per_sample = angular_error(pred, test_set["gaze"])
    # 입력을 보지 않는 하한선: 학습 데이터 평균 gaze
    mean_err = float(angular_error(np.broadcast_to(tr[2].mean(0), test_set["gaze"].shape),
                                   test_set["gaze"]).mean())
    result = {
        "test": test, "train": train_p, "val": val_p, "config": cfg,
        "test_error": test_err, "test_error_median": float(np.median(per_sample)),
        "mean_gaze_baseline": mean_err, "history": hist, "standardizer": std.state(),
    }
    if verbose:
        print(f"[fold {test}] test angular error {test_err:.2f}° (best epoch {hist['best_epoch']})", flush=True)
    if return_model:
        return result, net, std, pred
    return result


def save_result(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{result['test']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    return path


def load_results(out_dir):
    res = {}
    for p in PERSONS:
        path = os.path.join(out_dir, f"{p}.json")
        if os.path.exists(path):
            with open(path) as f:
                res[p] = json.load(f)
    return res
