import argparse
import pathlib

import torch

import disen


def train_joint_vae(
    dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path
) -> None:
    n_categories = 3
    n_continuous = 6
    gamma = 150.
    temperature = 0.67
    max_c_c = 1.1
    max_c_z = 40.
    max_c_at = 300_000
    lr = 0.0005
    batch_size = 64
    eval_batch_size = 1024
    n_epochs = 30

    dataset = disen.data.DSprites(dataset_path)
    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_categories + n_continuous, 1)
    model = disen.models.JointVAE(
        encoder,
        decoder,
        n_categories,
        n_continuous,
        gamma,
        temperature,
        max_c_c,
        max_c_z,
        max_c_at,
    )
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr)

    out_dir.mkdir()

    result = disen.training.train_model(
        model, dataset, optim, batch_size, eval_batch_size, n_epochs, out_dir
    )
    disen.evaluation.evaluate_mi_metrics_with_attacks(
        "jointvae", dataset, model, result, out_dir, noise=2.0, mix_rate=0.5
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
