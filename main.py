"""
Main entry point for running the analysis pipeline.
"""

import logging
from config.logging_config import logger
from examples.end_to_end_pipeline import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
