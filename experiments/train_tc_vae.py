import argparse
import pathlib

import torch

import disen


def train_tc_vae(
    dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path
) -> None:
    disen.setup_logger(out_dir)

    n_latents = 6

    dataset = disen.data.DSprites(dataset_path)
    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_latents, 1)
    model = disen.models.TCVAE(encoder, decoder, n_latents, len(dataset), beta=6.0)
    model.to(device)

    result = disen.training.train_model(
        model,
        dataset,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        batch_size=2048,
        eval_batch_size=1024,
        n_epochs=50,
        out_dir=out_dir,
    )
    disen.evaluation.evaluate_model(model, dataset, result, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    train_tc_vae(pathlib.Path(args.dataset), args.device, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
