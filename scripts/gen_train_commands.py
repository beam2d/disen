#!/usr/bin/env python3
import argparse
from typing import Optional, Sequence


def gen_train_commands(
    task: str,
    dataset_path: str,
    models: Sequence[str],
    betas: Sequence[Optional[float]],
    train_seeds: Sequence[int],
    out: str,
) -> None:
    for model in models:
        for beta in betas:
            for train_seed in train_seeds:
                cmd = [
                    "python3",
                    "scripts/train.py",
                    f"--out_dir={out}",
                    f"--dataset_path={dataset_path}",
                    f"--task={task}",
                    f"--model={model}",
                    f"--train_seed={train_seed}",
                ]
                if beta is not None:
                    cmd.append(f"--beta={beta}")
                print(*cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--beta")
    parser.add_argument("--train_seed", required=True)
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    betas: list[Optional[float]] = []
    if args.beta is None:
        betas.append(None)
    else:
        betas += map(float, args.beta.split(","))

    gen_train_commands(
        args.task,
        args.dataset_path,
        args.model.split(","),
        betas,
        list(map(int, args.train_seed.split(","))),
        args.out,
    )


if __name__ == "__main__":
    main()
