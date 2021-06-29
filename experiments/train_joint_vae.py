import argparse
import pathlib

import torch

import disen


def train_joint_vae(
    dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path
) -> None:
    n_categories = 3
    n_continuous = 6

    dataset = disen.data.DSprites(dataset_path)
    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_categories + n_continuous, 1)
    model = disen.models.JointVAE(
        encoder,
        decoder,
        n_categories=n_categories,
        n_continuous=n_continuous,
        gamma=150.,
        temperature=0.67,
        max_capacity_discrete=1.1,
        max_capacity_continuous=40.,
        max_capacity_iteration=300_000,
    )
    model.to(device)

    out_dir.mkdir()

    result = disen.training.train_model(
        model,
        dataset,
        optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
        batch_size=64,
        eval_batch_size=1024,
        n_epochs=30,
        out_dir=out_dir,
    )
    disen.evaluation.evaluate_mi_metrics_with_attacks(
        "jointvae", dataset, model, result, out_dir, alpha=[0.25, 0.5, 0.75, 1.0]
    )
    result.save(out_dir / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    train_joint_vae(pathlib.Path(args.dataset), args.device, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
