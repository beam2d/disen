import dataclasses
import json
import pathlib
import random
import subprocess
from typing import Literal, Optional

import numpy
import torch

from . import data, evaluation, models, nn, training


TaskType = Literal["dSprites"]
ModelType = Literal["betaVAE", "CascadeVAE", "FactorVAE", "JointVAE", "TCVAE"]


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

    def build_args(self) -> list[str]:
        args = [
            f"--out_dir={self.out_dir}",
            f"--dataset_path={self.dataset_path}",
            f"--task={self.task}",
            f"--model={self.model}",
            f"--train_seed={self.train_seed}",
        ]
        if self.phase == "eval":
            assert self.eval_seed is not None
            args.append(f"--eval_seed={self.eval_seed}")
            if self.alpha is not None:
                args.append(f"--alpha={self.alpha}")
        return args

    def get_model_path(self) -> pathlib.Path:
        return self._get_common_dir() / "phase-train" / "model.pt"

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

    def make_model(self) -> models.LatentVariableModel:
        if self.task == "dSprites":
            return _make_model_for_dsprites(self.model)
        raise ValueError(f"unknown task type: {self.task}")

    def save_run_info(self) -> None:
        info_dir = self.get_dir() / "job"
        info_dir.mkdir()

        with open(info_dir / "args.json", "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, default=str)
        with open(info_dir / "git.hash", "w") as f:
            subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)
        with open(info_dir / "git.diff", "w") as f:
            subprocess.run(["git", "diff"], stdout=f)

    def train(
        self, device: torch.device
    ) -> tuple[models.LatentVariableModel, evaluation.Result]:
        assert self.phase == "train"
        _init_seed(self.train_seed)
        model = self.make_model()
        exp_dir = self.get_dir()

        if self.task == "dSprites":
            result = _train_model_for_dsprites(
                model, self.model, self.dataset_path, device, exp_dir
            )
        else:
            raise ValueError(f"unknown task type: {self.task}")

        return model, result

    def _get_common_dir(self) -> pathlib.Path:
        out_dir = self.out_dir
        out_dir /= f"task-{self.task}"
        out_dir /= f"model-{self.model}"
        out_dir /= f"train_seed-{self.train_seed}"
        return out_dir


def _init_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def _make_model_for_dsprites(model: ModelType) -> models.LatentVariableModel:
    image_size = 64
    dataset_size = 737_280
    n_latents = 10
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
        return models.TCVAE(encoder, decoder, n_latents, dataset_size)

    raise ValueError(f"unknown model type: {model}")


def _train_model_for_dsprites(
    model: models.LatentVariableModel,
    model_type: ModelType,
    dataset_path: pathlib.Path,
    device: torch.device,
    out_dir: pathlib.Path,
) -> evaluation.Result:
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
