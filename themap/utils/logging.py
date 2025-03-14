import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

from themap.utils.config import LoggingConfig, default_logging_config


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
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(config.format, config.date_format))
    handlers.append(console_handler)
    
    # File handler if log_file is specified
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