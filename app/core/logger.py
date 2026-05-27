import sys
import logging
from logging.handlers import RotatingFileHandler
from app.core.config import settings


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d-%m-%Y %H:%M:%S")
    console_handler.setFormatter(console_format)

    try:
        file_handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s : %(lineno)d - %(message)s", datefmt="%d-%m-%Y %H:%M:%S")
        file_handler.setFormatter(file_format)
        
        logger.addHandler(file_handler)

    except Exception as e:
        print(f"Warning : Could not setup file logging - {e}")

    logger.addHandler(console_handler)

    return logger
