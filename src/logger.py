# logger.py
import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

def get_logger(name: str = "mlproject", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a configured logger instance.
    Logs to both console and a daily rotating file with 7-day backup

    Args:
        name (str): Logger name.
        log_dir (str): Directory to store log files.
        level (int): Logging level (e.g., Logging.INFO, logging.DEBUG).
    Returns:
        Logging.Logger: Configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Timestamped log file
    log_file=f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    log_file_path=os.path.join(log_dir, log_file)

    # Create logger
    logger=logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate=False # prevent duplicate logs

    # Only add handlers ones
    if not logger.handlers:
        # File handler with daily rotation
        file_handler=TimedRotatingFileHandler(
            log_file_path,
            when="midnight",
            backupCount=7,
            encoding="utf-8"
        )
        formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger