import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

from themap.utils.config import LoggingConfig, default_logging_config


class ColorizedFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output based on log level."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset to default
    }
    
    def __init__(self, fmt: str, datefmt: str, use_colors: bool = True):
        """
        Initialize the formatter with optional color support.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            use_colors: Whether to use colors in output
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors if enabled."""
        if self.use_colors and record.levelname in self.COLORS:
            # Save original levelname
            orig_levelname = record.levelname
            # Add color codes to levelname
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            # Format the record
            result = super().format(record)
            # Restore original levelname
            record.levelname = orig_levelname
            return result
        return super().format(record)


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Set up logging configuration for the THEMAP package.
    
    Args:
        config: Optional LoggingConfig instance. If None, uses default configuration.
    """
    if config is None:
        config = default_logging_config
    
    # Create logs directory if logging to file
    if config.log_file and config.create_log_dir:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up basic configuration
    handlers = []
    
    # Console handler with colorized output
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColorizedFormatter(
        config.format, 
        config.date_format,
        use_colors=config.use_colors
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler if log_file is specified (no colors for file output)
    if config.log_file:
        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.format, config.date_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level),
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Suppress verbose logging from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("rdkit").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {config.level}")
    if config.log_file:
        logger.info(f"Logging to file: {config.log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: The name of the logger (typically __name__ from the calling module)
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(name) 