"""Centralized logging configuration for GraphRAG system."""

import logging
import sys
from typing import Optional

from config import get_settings


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Console handler with rich formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Format: [LEVEL] module_name - message
    formatter = logging.Formatter(
        '[%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


class LoggingContext:
    """Context manager for structured logging with timing."""
    
    def __init__(self, logger: logging.Logger, message: str):
        self.logger = logger
        self.message = message
        self.start_time = None
    
    def __enter__(self):
        self.logger.info(f"Starting: {self.message}")
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"✓ Completed: {self.message} ({elapsed:.2f}s)")
        else:
            self.logger.error(
                f"✗ Failed: {self.message} ({elapsed:.2f}s)",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        return False  # Don't suppress exceptions
