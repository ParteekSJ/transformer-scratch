import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union


def setup_logger(
    name: str = "ai_research",
    log_dir: Optional[str] = "logs",
    log_level: Union[int, str] = logging.INFO,
    console_level: Union[int, str] = logging.INFO,
    file_level: Union[int, str] = logging.DEBUG,
) -> logging.Logger:
    """
    Set up a comprehensive logger for AI research projects.

    Args:
        name (str): Name of the logger
        log_dir (str, optional): Directory to store log files
        log_level (int/str): Overall logging level
        console_level (int/str): Logging level for console output
        file_level (int/str): Logging level for file output

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create logs directory if it doesn't exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Formatter with detailed information
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(filename)20s:%(lineno)3d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # File Handler (if log_dir is provided)
    if log_dir:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


# Example usage in an AI research context
def main():
    # Initialize logger
    logger = setup_logger(
        name="model_training", log_dir="./experiment_logs", console_level=logging.INFO, file_level=logging.DEBUG
    )

    try:
        # Simulated AI training process
        logger.info("Starting model training")
        logger.debug("Loading hyperparameters")

        # Hyperparameters
        learning_rate = 0.001
        batch_size = 64

        logger.info(f"Training with learning rate: {learning_rate}")
        logger.info(f"Batch size: {batch_size}")

        # Simulate some training steps
        for epoch in range(5):
            logger.debug(f"Epoch {epoch+1}/5")

            # Simulate potential warning or error
            if epoch == 3:
                logger.warning("Potential overfitting detected")

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
