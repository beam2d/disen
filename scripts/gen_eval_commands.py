#!/usr/bin/env python3
import argparse
from typing import Optional, Sequence

import disen


def gen_eval_commands(
    task: str,
    dataset_path: str,
    models: Sequence[str],
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    alphas: Sequence[Optional[float]],
    out: str,
) -> None:
    for model in models:
        for train_seed in train_seeds:
            for eval_seed in eval_seeds:
                for alpha in alphas:
                    print(
                        "python3",
                        "scripts/eval.py",
                        f"--out_dir={out}",
                        f"--dataset_path={dataset_path}",
                        f"--task={task}",
                        f"--model={model}",
                        f"--train_seed={train_seed}",
                        f"--eval_seed={eval_seed}",
                        f"--alpha={alpha}",
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_seed", required=True)
    parser.add_argument("--eval_seed", required=True)
    parser.add_argument("--alpha")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    gen_eval_commands(
        args.task,
        args.dataset_path,
        args.model.split(","),
        list(map(int, args.train_seed.split(","))),
        list(map(int, args.eval_seed.split(","))),
        list(map(disen.parse_optional(float), args.alpha.split(","))),
        args.out,
    )


if __name__ == "__main__":
    main()
