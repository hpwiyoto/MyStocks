import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")


def get_logger(name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
