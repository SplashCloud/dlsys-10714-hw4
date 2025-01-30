import logging
import sys
from typing import Optional

class NeedleLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with default settings"""
        self.logger = logging.getLogger('needle')
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self._file_handler = None
    
    def set_log_level(self, level: str):
        """Set the logging level.
        
        Args:
            level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    def add_file_handler(self, filename: str):
        """Add a file handler to also log to a file.
        
        Args:
            filename: Path to the log file
        """
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self._file_handler = logging.FileHandler(filename)
        self._file_handler.setFormatter(file_formatter)
        self.logger.addHandler(self._file_handler)
    
    def debug(self, msg: str):
        """Log a debug message"""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log an info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log a warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log an error message"""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Log a critical message"""
        self.logger.critical(msg)

# Global logger instance
global_logger = NeedleLogger()
global_logger.set_log_level('INFO')