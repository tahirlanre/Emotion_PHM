import logging
from functools import wraps

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} {result.shape}")
        return result

    return wrapper


def save_to_disk(dataf, path):
    dataf.to_csv(path, index=False)
