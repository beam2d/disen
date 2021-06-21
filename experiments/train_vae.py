import argparse
import pathlib

import torch
import tqdm

import disen


def train_vae(dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path) -> None:
    n_latents = 6
    beta = 5.0
    lr = 0.0005
    batch_size = 64
    n_epochs = 30

    dataset = disen.data.DSprites(dataset_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=1
    )

    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_latents, 1)
    model = disen.models.VAE(encoder, decoder, n_latents, beta)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr)

    out_dir.mkdir()

    for epoch in tqdm.tqdm(range(n_epochs)):
        model.train()
        for x, _ in loader:
            model.zero_grad(set_to_none=True)
            d = model(x.to(device))
            d["loss"].mean().backward()
            optim.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train_vae(args.dataset, args.device, args.out_dir)


if __name__ == "__main__":
    main()
