import argparse
import pathlib

import torch

import disen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_seed", required=True, type=int)
    parser.add_argument("--eval_seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=disen.parse_optional(float))
    args = parser.parse_args()

    experiment = disen.Experiment(
        out_dir=pathlib.Path(args.out_dir),
        dataset_path=pathlib.Path(args.dataset_path),
        task=args.task,
        model=args.model,
        train_seed=args.train_seed,
        phase="eval",
        eval_seed=args.eval_seed,
        alpha=args.alpha,
    )
    experiment.evaluate(torch.device(args.device))


if __name__ == "__main__":
    main()
