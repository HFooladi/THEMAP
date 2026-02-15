from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""

    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level: str = "WARNING"

    # Log file path (None for console-only logging)
    log_file: Optional[str] = None

    # Log format string (console)
    format: str = "%(levelname)s | %(name)s | %(message)s"

    # Date format string
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Whether to create log directory if it doesn't exist
    create_log_dir: bool = True

    # Log file rotation settings
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    # Whether to use colorized output for console logging
    use_colors: bool = True

    def __post_init__(self) -> None:
        """Validate and process configuration."""
        if self.log_file:
            self.log_file = str(Path(self.log_file))

        # Ensure log level is valid
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        self.level = self.level.upper()


# Verbose format for file logging
VERBOSE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"

# Default configuration
default_logging_config = LoggingConfig()
