import logging
import pathlib


def setup_logger(out_dir: pathlib.Path) -> None:
    out_dir.mkdir()
    logging.basicConfig(
        filename=str(out_dir / "log.txt"),
        filemode="w",
        format="[%(process)s|%(asctime)s(%(name)s)%(levelname)s] %(message)s",
        level=logging.INFO,
    )
