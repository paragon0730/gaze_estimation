"""GazeLeNet(numpy CNN)을 leave-one-person-out으로 학습·평가한다.

    python train_lopo.py                       # 15 fold 전부 → results/lopo/pXX.json
    python train_lopo.py --folds p00 p01       # 일부 fold만 (GitHub Actions에서 fold별 병렬 실행)
    python train_lopo.py --summarize           # pXX.json을 모아 results/lopo_results.csv 생성, 표 출력

하이퍼파라미터 기본값은 gaze/experiment.py의 CONFIG이며 모든 fold에 같게 쓴다.
"""
import argparse
import json
import os

import numpy as np

from gaze.data import CACHE_PATH, DATA_ROOT, PERSONS, load_all
from gaze.experiment import CONFIG, load_results, run_fold, save_result

# README의 baseline 결과 (ssuai/gaze_estimation)
RIDGE = [5.31, 7.76, 6.55, 8.05, 6.40, 7.35, 7.59, 7.97, 7.89, 8.51, 6.93, 6.63, 7.07, 7.85, 7.20]


def summarize(out_dir, csv_path):
    res = load_results(out_dir)
    if not res:
        raise SystemExit(f"{out_dir}에 결과가 없다")
    persons = [p for p in PERSONS if p in res]
    cnn = [res[p]["test_error"] for p in persons]
    mean_b = [res[p]["mean_gaze_baseline"] for p in persons]
    ridge = [RIDGE[PERSONS.index(p)] for p in persons]

    def row(name, vals, bold=False):
        m = f"{np.mean(vals):.2f}"
        return f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {'**' + m + '**' if bold else m} |"

    cfg = res[persons[0]]["config"]
    lines = [
        "| 방법 | " + " | ".join(persons) + " | 평균 |",
        "|------|" + "-----|" * len(persons) + "------|",
        row("평균 gaze 예측", mean_b),
        row("ridge 회귀 (baseline)", ridge),
        row("GazeLeNet (numpy CNN)", cnn, bold=True),
    ]
    best_epochs = [res[p]["history"]["best_epoch"] for p in persons]
    secs = [sum(res[p]["history"]["sec"]) for p in persons]
    notes = [
        "",
        f"- fold 수: {len(persons)}, 설정: `{json.dumps(cfg, ensure_ascii=False)}`",
        f"- 선택된 에폭(검증 사람 기준): {best_epochs}",
        f"- fold당 학습 시간: 평균 {np.mean(secs) / 60:.1f}분 (GitHub Actions ubuntu-latest, CPU numpy)",
    ]
    text = "\n".join(lines + notes) + "\n"
    with open(csv_path, "w") as f:
        f.write("person,cnn,ridge,mean_gaze,best_epoch\n")
        for p, c, r, m, e in zip(persons, cnn, ridge, mean_b, best_epochs):
            f.write(f"{p},{c:.4f},{r:.2f},{m:.4f},{e}\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DATA_ROOT)
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--folds", nargs="*", default=PERSONS)
    p.add_argument("--out", default="results/lopo")
    p.add_argument("--save-params", default=None, help="이 폴더에 fold별 학습된 매개변수 저장")
    p.add_argument("--summarize", action="store_true")
    for k in ("epochs", "batch_size", "lr", "lr_decay", "dropout_ratio", "seed"):
        p.add_argument("--" + k.replace("_", "-"), type=type(CONFIG[k]), default=None)
    p.add_argument("--no-head", action="store_true")
    args = p.parse_args()

    if args.summarize:
        summarize(args.out, "results/lopo_results.csv")
        return

    cfg = {k: getattr(args, k) for k in ("epochs", "batch_size", "lr", "lr_decay", "dropout_ratio", "seed")
           if getattr(args, k) is not None}
    if args.no_head:
        cfg["use_head"] = False
    data = load_all(args.root, args.cache)
    for test in args.folds:
        result, net, std, _ = run_fold(data, test, cfg, return_model=True)
        print("saved", save_result(result, args.out))
        if args.save_params:
            os.makedirs(args.save_params, exist_ok=True)
            net.save_params(os.path.join(args.save_params, f"{test}.pkl"), extra={"standardizer": std.state()})


if __name__ == "__main__":
    main()
