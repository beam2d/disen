import contextlib
import logging
import pathlib
from typing import Iterator

import torch


def setup_logger(out_dir: pathlib.Path) -> None:
    logging.basicConfig(
        filename=str(out_dir / "log.txt"),
        filemode="w",
        format="[%(process)s|%(asctime)s(%(name)s)%(levelname)s] %(message)s",
        level=logging.INFO,
    )


@contextlib.contextmanager
def torch_sci_mode_disabled() -> Iterator[None]:
    torch.set_printoptions(sci_mode=False)
    try:
        yield
    finally:
        torch.set_printoptions()
