import logging
import pathlib


def setup_logger(out_dir: pathlib.Path) -> None:
    logging.basicConfig(
        filename=str(out_dir / "log.txt"),
        filemode="w",
        format="[%(process)s|%(asctime)s(%(name)s)%(levelname)s] %(message)s",
        level=logging.INFO,
    )
