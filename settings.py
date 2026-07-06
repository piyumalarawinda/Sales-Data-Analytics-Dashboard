"""
Logging configuration for the project.
"""

import logging
import logging.config
from config.settings import LOG_LEVEL, LOG_FORMAT

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': LOG_FORMAT
        },
    },
    'handlers': {
        'console': {
            'level': LOG_LEVEL,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'level': LOG_LEVEL,
            'class': 'logging.FileHandler',
            'filename': 'logs/app.log',
            'formatter': 'standard',
        },
    },
    'root': {
        'level': LOG_LEVEL,
        'handlers': ['console', 'file'],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
