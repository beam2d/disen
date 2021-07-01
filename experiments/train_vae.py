import argparse
import pathlib

import torch

import disen


def train_vae(dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path) -> None:
    n_latents = 6

    dataset = disen.data.DSprites(dataset_path)
    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_latents, 1)
    model = disen.models.VAE(encoder, decoder, n_latents, beta=5.0)
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
    disen.evaluation.render_latent_traversal(dataset, model, 12, out_dir / "traversal")
    disen.evaluation.evaluate_factor_vae_score(model, dataset, result)
    disen.evaluation.evaluate_beta_vae_score(model, dataset, result, out_dir)
    disen.evaluation.evaluate_mi_metrics_with_attacks(
        "vae", dataset, model, result, out_dir, alpha=[0.25, 0.5, 0.75, 1.0]
    )
    result.save(out_dir / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    train_vae(pathlib.Path(args.dataset), args.device, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
