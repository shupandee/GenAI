"""Logging configuration for the trading bot."""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Configure logging for the trading bot.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler - logs everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger