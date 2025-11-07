import logging
import sys
import warnings

import coloredlogs


def create_logger(name, enable_file_logging=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.flush = sys.stderr.flush

    if enable_file_logging:
        file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (%(name)s) %(message)s"))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    logger.addHandler(console_handler)
    coloredlogs.install(
        fmt="%(asctime)s %(levelname)s %(message)s",
        field_styles={"asctime": {"color": "blue"}, "levelname": {"color": "yellow", "bold": True}},
        level="DEBUG",
        logger=logger,
    )
    logger.propagate = False
    return logger


logger = create_logger("suptoolbox")


warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(
    "ignore",
)
warnings.filterwarnings(
    "ignore",
    message=".*is deprecated.*",
)
