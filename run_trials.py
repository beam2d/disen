#!/usr/bin/env python3
"""Run experiments multiple times.

It executes the following command for ``n`` in ``range(n_trials)``.

```
$ python3 {script} --device={device} --out_dir={out_dir/n} {args}
```

The script must save ``disen.experiment.Result`` object to ``out_dir/n/result.json``.
This multi-run script collects them and make a summary.

"""
import argparse
import concurrent.futures
import pathlib
import queue
import subprocess
import torch

import disen


def run_trials(
    script: str, out_dir: pathlib.Path, n_trials: int, args: list[str]
) -> None:
    device_q: queue.Queue[str] = queue.Queue()
    for n in range(torch.cuda.device_count()):
        device_q.put(f"cuda:{n}")

    def trial(n: int) -> disen.evaluation.Result:
        device = device_q.get()
        try:
            trial_dir = out_dir / str(n)
            subprocess.run(
                ["python3", script, "--device", device, "--out_dir", trial_dir, *args]
            )
        finally:
            device_q.put(device)
        return disen.evaluation.Result.load(trial_dir / "result.json")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(trial, range(n_trials)))

    disen.evaluation.summarize_multiple_trials(results, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser("Run experiments multiple times")
    parser.add_argument("script", type=str)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=8)
    args, extra = parser.parse_known_args()

    run_trials(args.script, pathlib.Path(args.out_dir), args.n_trials, extra)


if __name__ == "__main__":
    main()
