import dataclasses
import json
import pathlib
import random
import subprocess
from typing import Any, Callable, Iterator, Literal, Optional, TypeVar, cast

import numpy
import torch

from . import attack, data, evaluation, log_util, models, nn, training


TaskType = Literal["dSprites"]
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

    def get_dir(self) -> pathlib.Path:
        out_dir = self._get_common_dir()
        out_dir /= f"phase-{self.phase}"
        if self.eval_seed is not None:
            out_dir /= f"eval_seed-{self.eval_seed}"
        if self.alpha is not None:
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

        if self.task == "dSprites":
            history = _train_model_for_dsprites(
                model, self.model, self.dataset_path, device, exp_dir
            )
        else:
            raise ValueError(f"unknown task type: {self.task}")

        torch.save(model.state_dict(), self.get_model_path())
        history.save(exp_dir / "history.json")

    def evaluate(self, device: torch.device) -> None:
        assert self.phase == "eval"
        assert self.eval_seed is not None

        self._setup_exp_dir()
        _init_seed(self.eval_seed)
        model = self._make_model()
        pt_path = self.get_model_path()
        model.load_state_dict(torch.load(pt_path))
        entry = _evaluate_model_for_dsprites(
            model, self.dataset_path, device, self.alpha, self.get_dir()
        )
        with open(self.get_entry_path(), "w") as f:
            json.dump(entry, f, indent=4)

    def has_entry(self) -> bool:
        return self.get_entry_path().exists()

    def load_entry(self) -> dict[str, float]:
        with open(self.get_entry_path()) as f:
            return json.load(f)

    def load_entry_with_attrs(self) -> dict[str, Any]:
        entry: dict[str, Any] = self.load_entry()
        assert self.eval_seed is not None
        assert self.alpha is not None
        entry.update(
            {
                "task": self.task,
                "model": self.model,
                "train_seed": self.train_seed,
                "eval_seed": self.eval_seed,
                "alpha": self.alpha,
            }
        )
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
        if self.task == "dSprites":
            return _make_model_for_dsprites(self.model)
        raise ValueError(f"unknown task type: {self.task}")


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
                for alpha, d in dig(d, "alpha", float):
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


def _make_model_for_dsprites(model: ModelType) -> models.LatentVariableModel:
    image_size = 64
    dataset_size = 737_280
    n_latents = 6
    n_categories = 3
    n_continuous = n_latents - 1
    n_latent_features = n_categories + n_continuous

    if model == "betaVAE":
        encoder = nn.SimpleConvNet(image_size, 1, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latents, 1)
        return models.VAE(encoder, decoder, n_latents, beta=4.0)

    if model == "CascadeVAE":
        encoder = nn.SimpleConvNet(image_size, 1, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latent_features, 1)
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
        encoder = nn.SimpleConvNet(image_size, 1, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latents, 1)
        discr = models.FactorVAEDiscriminator(n_latents)
        return models.FactorVAE(encoder, decoder, discr, gamma=35.0)

    if model == "JointVAE":
        encoder = nn.SimpleConvNet(image_size, 1, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latent_features, 1)
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
        encoder = nn.SimpleConvNet(image_size, 1, 256)
        decoder = nn.SimpleTransposedConvNet(image_size, n_latents, 1)
        return models.TCVAE(encoder, decoder, n_latents, dataset_size, beta=6.0)

    raise ValueError(f"unknown model type: {model}")


def _train_model_for_dsprites(
    model: models.LatentVariableModel,
    model_type: ModelType,
    dataset_path: pathlib.Path,
    device: torch.device,
    out_dir: pathlib.Path,
) -> training.History:
    dataset = data.DSprites(dataset_path)
    model.to(device)

    if model_type == "betaVAE":
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
            batch_size=64,
            eval_batch_size=2048,
            n_epochs=50,
            out_dir=out_dir,
        )

    if model_type == "CascadeVAE":
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
            batch_size=64,
            eval_batch_size=2048,
            n_epochs=50,
            out_dir=out_dir,
        )

    if model_type == "FactorVAE":
        assert isinstance(model, models.FactorVAE)
        return training.train_factor_vae(
            model,
            dataset,
            batch_size=64,
            eval_batch_size=2048,
            lr=1e-4,
            betas=(0.9, 0.999),
            d_lr=1e-4,
            d_betas=(0.5, 0.9),
            n_epochs=50,
            out_dir=out_dir,
        )

    if model_type == "JointVAE":
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
            batch_size=64,
            eval_batch_size=2048,
            n_epochs=50,
            out_dir=out_dir,
        )

    if model_type == "TCVAE":
        return training.train_model(
            model,
            dataset,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            batch_size=2048,
            eval_batch_size=2048,
            n_epochs=50,
            out_dir=out_dir,
        )

    raise TypeError(f"unknown model type: {type(model)}")


def _evaluate_model_for_dsprites(
    model: models.LatentVariableModel,
    dataset_path: pathlib.Path,
    device: torch.device,
    alpha: Optional[float],
    out_dir: pathlib.Path,
) -> dict[str, float]:
    dataset = data.DSprites(dataset_path)
    model.to(device)

    alpha = alpha or 0.0
    if alpha != 0.0:
        D = model.spec.real_components_size
        U = (torch.eye(D) - 2 / D).to(model.device)
        model = attack.RedundancyAttack(model, alpha, U)

    entry: dict[str, float] = {}

    # entry["unibound_l_dre"] = evaluation.unibound_lower(model, dataset)
    # entry["unibound_u_dre"] = evaluation.unibound_upper(model, dataset)
    entry.update(evaluation.estimate_unibound_in_many_ways(model, dataset, out_dir))

    entry["factor_vae_score"] = evaluation.factor_vae_score(model, dataset)
    entry["beta_vae_score"] = evaluation.beta_vae_score(model, dataset)

    if model.has_valid_elemwise_posterior:
        mi = evaluation.mi_metrics(model, dataset)
        mi.save(out_dir / "mi_metrics.txt")
        entry.update(mi.get_scores())

    return entry
