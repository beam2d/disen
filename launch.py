#!/usr/bin/env python3
import argparse
import datetime
import os
import pathlib
import subprocess
import sys


def main() -> None:
    group = os.getenv("GROUP") or exit(1)
    user = os.getenv("USER") or exit(1)
    lustre = pathlib.Path("/lustre") / group / user
    expdir = lustre / "src" / "github.com" / "beam2d" / "disen"

    parser = argparse.ArgumentParser()
    parser.add_argument("jobtype", choices=("h", "l"))
    parser.add_argument("walltime")
    parser.add_argument("--list", "-l")
    parser.add_argument("--cmd", "-c")
    parser.add_argument("--job_out", "-o", default=str(expdir / "tmp" / "jobout"))
    parser.add_argument("--max_jobs", "-j", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dt = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    job_out = pathlib.Path(f"{args.job_out}_{dt}")
    job_out.mkdir()

    def launch_one(cmd: str, sh_path: pathlib.Path) -> None:
        script = f"""#!/bin/bash
#PBS -q {args.jobtype}-regular
#PBS -l select=1:mpiprocs=1:ompthreads=4
#PBS -W group_list={group}
#PBS -l walltime={args.walltime}:00
#PBS -o {sh_path}.stdout
#PBS -e {sh_path}.stderr

export HOME={lustre}
. $HOME/env.sh
cd {expdir}
. venv/bin/activate
export PYTHONPATH=.
{cmd}
"""
        with open(sh_path, "w") as f:
            f.write(script)
        if args.dry_run:
            print("qsub", sh_path)
        else:
            subprocess.run(["qsub", sh_path], check=True, stderr=subprocess.STDOUT)

    if args.cmd:
        launch_one(args.cmd, job_out / "launch.sh")
        return

    n_gpus = {"h": 2, "l": 4}[args.jobtype]

    assert args.list
    with open(args.list) as f:
        cmds = [cmd.strip() for cmd in f.readlines()]

    multigpu_cmds: list[list[str]] = []
    while cmds:
        new, cmds = cmds[:n_gpus], cmds[n_gpus:]
        multigpu_cmds.append(new)
    multigpu_cmds.reverse()

    n_jobs = min(args.max_jobs or 9, len(multigpu_cmds))
    jobs: list[list[list[str]]] = [[] for _ in range(n_jobs)]
    while multigpu_cmds:
        for j in jobs:
            if not multigpu_cmds:
                break
            j.append(multigpu_cmds.pop())
    jobs = [j for j in jobs if j]

    count = 0

    def launch_multi(cmds: list[list[str]]) -> None:
        assert all(len(cmd) <= n_gpus for cmd in cmds)
        nonlocal count
        sh_path = job_out / f"launch{count}.sh"

        long_cmds: list[str] = []
        for cmd_gpus in cmds:
            cmds_with_gpu: list[str] = []
            for i, cmd in enumerate(cmd_gpus):
                p = min(cmd.find("<"), cmd.find(">"))
                p = p if p >= 0 else len(cmd)
                cmds_with_gpu.append(cmd[:p] + f" --device=cuda:{i} " + cmd[p:])
            long_cmd = "time " + " & ".join(cmds_with_gpu) + " && wait\n"
            long_cmds.append(long_cmd)

        launch_one("".join(long_cmds), sh_path)

        for cmd_gpus in cmds:
            for cmd in cmd_gpus:
                print(f"{sh_path}: {cmd}", file=sys.stderr)

        count += 1

    remained_jobs: list[list[str]] = []
    for job in jobs:
        try:
            launch_multi(job)
        except subprocess.CalledProcessError as e:
            print(e.stdout, file=sys.stderr)
            remained_jobs += job
            break

    if remained_jobs:
        print("=====", file=sys.stderr)
        for remained_job in remained_jobs:
            print("\n".join(remained_job), file=sys.stderr)


if __name__ == "__main__":
    main()
