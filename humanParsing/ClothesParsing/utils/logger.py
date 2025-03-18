import os
import logging
import datetime
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """Logging to Tensorboard."""

    def __init__(
            self,
            log_dir: str,
            filename_suffix: Optional[str] = None
    ):
        """
        Initialize TensorboardLogger.

        Args:
            log_dir (str): Directory to save logs
            filename_suffix (str, optional): Suffix for log filename
        """
        if filename_suffix:
            log_dir = os.path.join(
                log_dir,
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename_suffix}"
            )
        else:
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """Log image to tensorboard.

        Args:
            tag (str): Image tag
            image (Union[torch.Tensor, np.ndarray]): Image to log
            step (int): Global step
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.ndim == 3:
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)

        # Ensure image is in correct format (HWC)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected image in HWC format with 3 channels, got shape {image.shape}")

        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        self.writer.add_image(tag, image, step, dataformats='HWC')

    def log_figures(self, tag: str, figures, step: int):
        """Log matplotlib figures."""
        self.writer.add_figure(tag, figures, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)

    def close(self):
        """Close writer."""
        self.writer.close()


class Logger:
    """General purpose logger with console and file output."""

    def __init__(
            self,
            name: str,
            save_dir: str,
            filename: str = "log.txt",
            level: int = logging.INFO
    ):
        """
        Initialize Logger.

        Args:
            name (str): Logger name
            save_dir (str): Directory to save log file
            filename (str): Log filename
            level (int): Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create handlers
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_dir, filename))
        console_handler = logging.StreamHandler()

        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)