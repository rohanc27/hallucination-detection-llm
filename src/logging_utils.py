import logging
import sys
from pathlib import Path
from datetime import datetime

LOGGER_NAME = "hallucination_detection"


def setup_logging(log_dir="results/logs", log_level=logging.INFO, run_id=None):
    """Initialize logging to file + console with timestamps. Returns (logger, log_file, run_id)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or timestamp
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s:%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    logger.info(f"Run ID: {run_id}")
    return logger, log_file, run_id


if __name__ == "__main__":
    logger, path, run_id = setup_logging()
    logger.info("Test log message")
    print(f"Log saved to: {path}")
