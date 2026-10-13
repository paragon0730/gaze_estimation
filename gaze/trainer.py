"""미니배치 학습 루프. 교재 4장 train_neuralnet / common/trainer.py와 같은 흐름에
회귀용 평가(각도 오차)와 검증 기반 모델 선택을 더했다."""
import time

import numpy as np

from common.optimizer import SGD, Momentum, AdaGrad, Adam, RMSprop
from gaze.metrics import angular_error

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adagrad": AdaGrad, "adam": Adam, "rmsprop": RMSprop}


def evaluate(network, std, x, head, gaze):
    """원래 단위(라디안)로 되돌린 예측과 정답의 평균 각도 오차(°)."""
    pred = std.inverse_t(network.predict_batches(x, head))
    return float(angular_error(pred, gaze).mean()), pred


def train(network, std, train_set, val_set=None, epochs=10, batch_size=128, optimizer="adam",
          lr=1e-3, lr_decay=1.0, seed=0, log_every=100, verbose=True):
    """
    train_set, val_set: 표준화된 (x, head, t)와 원래 단위 gaze를 묶은 dict
        {"x": (N,1,36,60), "head": (N,2), "t": (N,2), "gaze": (N,2)}
    val_set이 있으면 에폭마다 검증 각도 오차를 재고, 가장 낮은 에폭의 매개변수를 최종으로 쓴다.
    lr_decay: 에폭이 끝날 때마다 lr에 곱한다.
    """
    rng = np.random.default_rng(seed)
    opt = OPTIMIZERS[optimizer](lr=lr)
    n = len(train_set["x"])
    iters_per_epoch = n // batch_size

    hist = {"iter_loss": [], "epoch_loss": [], "train_err": [], "val_err": [], "lr": [], "sec": []}
    best = (np.inf, None, -1)
    # 학습 오차는 고정된 일부 샘플로 잰다 (교재 Trainer의 evaluate_sample_num_per_epoch와 같은 이유)
    probe = rng.choice(n, size=min(n, 3000), replace=False)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(n)
        losses = []
        for it in range(iters_per_epoch):
            idx = order[it * batch_size:(it + 1) * batch_size]
            loss, grads = network.gradient(train_set["x"][idx], train_set["head"][idx], train_set["t"][idx])
            opt.update(network.params, grads)
            losses.append(float(loss))
            if verbose and log_every and (it + 1) % log_every == 0:
                print(f"    epoch {epoch} iter {it + 1}/{iters_per_epoch} loss {np.mean(losses[-log_every:]):.4f}",
                      flush=True)
        hist["iter_loss"].extend(losses)
        hist["epoch_loss"].append(float(np.mean(losses)))
        hist["lr"].append(opt.lr)

        train_err, _ = evaluate(network, std, train_set["x"][probe], train_set["head"][probe],
                                train_set["gaze"][probe])
        hist["train_err"].append(train_err)
        msg = f"  epoch {epoch:2d}  loss {hist['epoch_loss'][-1]:.4f}  train {train_err:.2f}°"
        if val_set is not None:
            val_err, _ = evaluate(network, std, val_set["x"], val_set["head"], val_set["gaze"])
            hist["val_err"].append(val_err)
            msg += f"  val {val_err:.2f}°"
            if val_err < best[0]:
                best = (val_err, network.copy_params(), epoch)
        hist["sec"].append(time.time() - t0)
        if verbose:
            print(msg + f"  ({hist['sec'][-1]:.0f}s)", flush=True)
        opt.lr *= lr_decay

    if best[1] is not None:
        network.set_params(best[1])
        hist["best_epoch"] = best[2]
    else:
        hist["best_epoch"] = epochs
    return hist
