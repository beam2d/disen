import dataclasses
import json
import pathlib
import random
import subprocess
from typing import Any, Callable, Iterator, Literal, Optional, TypeVar, cast

import numpy
import torch

from . import attack, data, evaluation, log_util, models, nn, str_util, training


TaskType = Literal["dSprites", "3dshapes"]
ModelType = Literal["betaVAE", "CascadeVAE", "FactorVAE", "JointVAE", "TCVAE"]
_T = TypeVar("_T")


@dataclasses.dataclass
class Experiment:
    out_dir: pathlib.Path
    dataset_path: pathlib.Path
    task: TaskType
    model: ModelType
    train_seed: int
    phase: Literal["train", "eval"]
    eval_seed: Optional[int] = None
    alpha: Optional[float] = None

    def get_model_path(self) -> pathlib.Path:
        return self._get_common_dir() / "phase-train" / "model.pt"

    def get_entry_path(self) -> pathlib.Path:
        assert self.phase == "eval"
        return self.get_dir() / "entry.json"

    def get_history_path(self) -> pathlib.Path:
        return self._get_common_dir() / "phase-train" / "history.json"

    def get_dir(self) -> pathlib.Path:
        out_dir = self._get_common_dir()
        out_dir /= f"phase-{self.phase}"
        if self.eval_seed is not None:
            out_dir /= f"eval_seed-{self.eval_seed}"
            out_dir /= f"alpha-{self.alpha}"
        return out_dir

    def get_job_dir(self) -> pathlib.Path:
        return self.get_dir() / "job"

    def train(self, device: torch.device) -> None:
        assert self.phase == "train"

        self._setup_exp_dir()
        _init_seed(self.train_seed)
        model = self._make_model()
        exp_dir = self.get_dir()

        history = _train_model(
            self.task, model, self.model, self.dataset_path, device, exp_dir
        )

        torch.save(model.state_dict(), self.get_model_path())
        history.save(exp_dir / "history.json")

    def evaluate(self, device: torch.device) -> None:
        assert self.phase == "eval"
        assert self.eval_seed is not None

        self._setup_exp_dir()
        _init_seed(self.eval_seed)
        model = self._make_model()
        pt_path = self.get_model_path()
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        entry = _evaluate_model(
            self.task, model, self.dataset_path, device, self.alpha, self.get_dir()
        )
        with open(self.get_entry_path(), "w") as f:
            json.dump(entry, f, indent=4)

    def has_entry(self) -> bool:
        return self.get_entry_path().exists()

    def load_entry(self) -> dict[str, float]:
        with open(self.get_history_path()) as f:
            history: list[dict[str, float]] = json.load(f)
            d = history[-1]
        with open(self.get_entry_path()) as f:
            entry: dict[str, float] = json.load(f)
        entry.update(d)
        return entry

    def load_entry_with_attrs(self) -> dict[str, Any]:
        entry: dict[str, Any] = self.load_entry()
        entry["task"] = self.task
        entry["model"] = self.model
        entry["train_seed"] = self.train_seed
        if self.eval_seed is not None:
            entry["eval_seed"] = self.eval_seed
        entry["alpha"] = self.alpha
        return entry

    def _get_common_dir(self) -> pathlib.Path:
        out_dir = self.out_dir
        out_dir /= f"task-{self.task}"
        out_dir /= f"model-{self.model}"
        out_dir /= f"train_seed-{self.train_seed}"
        return out_dir

    def _setup_exp_dir(self) -> None:
        exp_dir = self.get_dir()
        exp_dir.mkdir(parents=True)
        log_util.setup_logger(exp_dir)

        info_dir = self.get_dir() / "job"
        info_dir.mkdir()

        with open(info_dir / "args.json", "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, default=str)
        with open(info_dir / "git.hash", "w") as f:
            subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)
        with open(info_dir / "git.diff", "w") as f:
            subprocess.run(["git", "diff"], stdout=f)

    def _make_model(self) -> models.LatentVariableModel:
        return _make_model(self.task, self.model)


def collect_experiments(
    root: pathlib.Path,
    task: TaskType,
    dataset_path: pathlib.Path = pathlib.Path("."),
) -> Iterator[Experiment]:
    def dig(
        d: pathlib.Path, attr: str, typ: Callable[[str], _T]
    ) -> list[tuple[_T, pathlib.Path]]:
        ret: list[tuple[_T, pathlib.Path]] = []
        if not d.exists():
            return ret
        for l in d.iterdir():
            kv = l.name.split("-")
            if len(kv) == 2 and kv[0] == attr:
                ret.append((typ(kv[1]), l))
        return ret

    for model, d in dig(root / f"task-{task}", "model", str):
        for train_seed, d in dig(d, "train_seed", int):
            for eval_seed, d in dig(d / "phase-eval", "eval_seed", int):
                for alpha, _ in dig(d, "alpha", str_util.parse_optional(float)):
                    yield Experiment(
                        root,
                        dataset_path,
                        task,
                        cast(ModelType, model),
                        train_seed,
                        "eval",
                        eval_seed,
                        alpha,
                    )


def _init_seed(seed: int) -> None:
    seed *= 1_091
    random.seed(seed + 12)
    numpy.random.seed(seed + 12_345)
    torch.manual_seed(seed + 12_345_678)


def _get_dataset(task: TaskType, path: pathlib.Path) -> data.DatasetWithFactors:
    if task == "dSprites":
        return data.DSprites(path)
    if task == "3dshapes":
        return data.Shapes3d(path)
    raise ValueError(f"unknown task: {task}")


def _make_model(task: TaskType, model: ModelType) -> models.LatentVariableModel:
    if task == "dSprites":
        image_size = 64
        channels = 1
        dataset_size = 737_280
        n_latents = 6
        n_categories = 3
    elif task == "3dshapes":
        image_size = 64
        channels = 3
        dataset_size = 480_000
        n_latents = 6
        n_categories = 4
    else:
        raise ValueError(f"unknown task: {task}")

    n_continuous = n_latents - 1
    n_latent_features = n_categories + n_continuous

    if model == "betaVAE":
        encoder = nn.SimpleConvNet(image_size, channels, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latents, channels)
        return models.VAE(encoder, decoder, n_latents, beta=4.0)

    if model == "CascadeVAE":
        encoder = nn.SimpleConvNet(image_size, channels, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latent_features, channels)
        return models.CascadeVAE(
            encoder,
            decoder,
            n_categories=n_categories,
            n_continuous=n_continuous,
            beta_h=10.0,
            beta_l=2.0,
            beta_dumping_interval=20_000,
            warmup_iteration=100_000,
            duplicate_penalty=0.001,
        )

    if model == "FactorVAE":
        encoder = nn.SimpleConvNet(image_size, channels, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latents, channels)
        discr = models.FactorVAEDiscriminator(n_latents)
        gamma = {"dSprites": 35.0, "3dshapes": 20.0}[task]
        return models.FactorVAE(encoder, decoder, discr, gamma=gamma)

    if model == "JointVAE":
        encoder = nn.SimpleConvNet(image_size, channels, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latent_features, channels)
        return models.JointVAE(
            encoder,
            decoder,
            n_categories=n_categories,
            n_continuous=n_continuous,
            gamma=150.0,
            temperature=0.67,
            max_capacity_discrete=1.1,
            max_capacity_continuous=40.0,
            max_capacity_iteration=300_000,
        )

    if model == "TCVAE":
        beta = {"dSprites": 6.0, "3dshapes": 4.0}[task]
        encoder = nn.SimpleConvNet(image_size, channels, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latents, channels)
        return models.TCVAE(encoder, decoder, n_latents, dataset_size, beta=beta)

    raise ValueError(f"unknown model: {model}")


def _train_model(
    task: TaskType,
    model: models.LatentVariableModel,
    model_type: ModelType,
    dataset_path: pathlib.Path,
    device: torch.device,
    out_dir: pathlib.Path,
) -> training.History:
    dataset = _get_dataset(task, dataset_path)
    n_iters = {"dSprites": 300_000, "3dshapes": 500_000}[task]
    batch_size = 64
    eval_batch_size = 2048
    model.to(device)

    if model_type == "betaVAE":
        lr = {"dSprites": 5e-4, "3dshapes": 1e-4}[task]
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            n_iters=n_iters,
            out_dir=out_dir,
        )

    if model_type == "CascadeVAE":
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            n_iters=n_iters,
            out_dir=out_dir,
        )

    if model_type == "FactorVAE":
        assert isinstance(model, models.FactorVAE)
        d_lr = {"dSprites": 1e-4, "3dshapes": 1e-5}[task]
        return training.train_factor_vae(
            model,
            dataset,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr=1e-4,
            betas=(0.9, 0.999),
            d_lr=d_lr,
            d_betas=(0.5, 0.9),
            n_iters=n_iters,
            out_dir=out_dir,
        )

    if model_type == "JointVAE":
        lr = {"dSprites": 5e-4, "3dshapes": 1e-4}[task]
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            n_iters=n_iters,
            out_dir=out_dir,
        )

    if model_type == "TCVAE":
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            batch_size=2048,
            eval_batch_size=eval_batch_size,
            n_iters=n_iters // 10,
            out_dir=out_dir,
        )

    raise ValueError(f"unknown model type: {model_type}")


def _evaluate_model(
    task: TaskType,
    model: models.LatentVariableModel,
    dataset_path: pathlib.Path,
    device: torch.device,
    alpha: Optional[float],
    out_dir: pathlib.Path,
) -> dict[str, float]:
    dataset = _get_dataset(task, dataset_path)
    model.to(device)

    if alpha is not None:
        D = model.spec.real_components_size
        U = (torch.eye(D) - 2 / D).to(model.device)
        model = attack.RedundancyAttack(model, alpha, U)

    entry: dict[str, float] = {}

    entry["factor_vae_score"] = evaluation.factor_vae_score(model, dataset)
    entry["beta_vae_score"] = evaluation.beta_vae_score(model, dataset)

    if model.has_valid_elemwise_posterior:
        mi = evaluation.mi_metrics(model, dataset)
        mi.save(out_dir)
        entry.update(mi.get_scores())

    return entry
