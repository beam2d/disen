import argparse
import pathlib

import torch

import disen


def evaluate(experiment: disen.Experiment, device: torch.device) -> None:
    exp_dir = experiment.get_dir()
    exp_dir.mkdir(parents=True)
    disen.setup_logger(exp_dir)

    experiment.save_run_info()
    entry = experiment.evaluate(device)
    entry.save(exp_dir / "entry.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_seed", required=True, type=int)
    parser.add_argument("--eval_seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.0)
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
    evaluate(experiment, torch.device(args.device))


if __name__ == "__main__":
    main()
