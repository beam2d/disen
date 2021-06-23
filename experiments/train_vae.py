import argparse
import pathlib
from typing import Callable

import torch

import disen


def train_vae(dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path) -> None:
    n_latents = 6
    beta = 5.0
    lr = 0.0005
    batch_size = 64
    eval_batch_size = 1024
    n_epochs = 30

    dataset = disen.data.DSprites(dataset_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=1
    )
    eval_loader = torch.utils.data.DataLoader(dataset, eval_batch_size, num_workers=1)

    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_latents, 1)
    model = disen.models.VAE(encoder, decoder, n_latents, beta)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr)

    out_dir.mkdir()
    result = disen.evaluation.Result([], {})
    iteration = 0

    for epoch in range(n_epochs):
        print(f"epoch: {epoch + 1}...")
        model.train()
        for x, _ in loader:
            model.zero_grad(set_to_none=True)
            d = model(x.to(device))
            d["loss"].mean().backward()
            optim.step()
            iteration += 1

        model.eval()
        with torch.no_grad():
            accum = disen.evaluation.BatchAccumulator()
            for x, _ in eval_loader:
                d = model(x.to(device))
                accum.accumulate(d)

            entry = accum.mean()
            entry["iteration"] = iteration
            result.history.append(entry)

        result.plot_history(out_dir)

    attacks = {
        "vae": model,
        "vae_noised": disen.attack.NoisedCopyAttack(model, 2.0),
        "vae_mixed": disen.attack.GlobalMixingAttack(model, 0.5),
    }
    for name, target in attacks.items():
        mi_metrics = disen.evaluation.evaluate_mi_metrics(dataset, target)
        mi_metrics.save(out_dir / f"mi_metrics-{name}.txt")
        mi_metrics.set_final_metrics(result.final_metrics, name)

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
