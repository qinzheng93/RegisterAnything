import sys

import loguru

from .distributed import is_master


class Logger:
    """Advanced logger with stderr and log file, based on loguru.

    Note:
        When DDP is enabled, only WARNING and ERROR logs are activated for slave processes. File logger is only enabled on the master process.
    """

    def __init__(self):
        self.logger = loguru.logger
        self.logger.remove()

    def initialize(self, use_stderr: bool = True, log_file: str | None = None) -> None:
        # Set log level and log file
        log_level = "DEBUG" if is_master() else "WARNING"

        # add stderr logger
        if use_stderr:
            fmt_str = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level><n>{message}</n></level>"
            )
            self.logger.add(sys.stderr, format=fmt_str, colorize=True, level=log_level)

        # add file logger
        if is_master() and log_file is not None:
            fmt_str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            self.logger.add(log_file, format=fmt_str, level="INFO")

    def log(self, message: str, level: str = "INFO") -> None:
        if level not in ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            self.logger.warning(f"Unsupported logging level: {level}. Fallback to INFO.")
            level = "INFO"
        self.logger.log(level, message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def success(self, message: str) -> None:
        self.logger.success(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


logger = Logger()
