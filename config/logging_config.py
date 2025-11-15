"""
Configure logging for both gunicorn and application.
"""

import os

from dotenv import load_dotenv

load_dotenv()

logging_cfg = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'access': {
            'format': (
                "[%(asctime)s]:[%(levelname)s]:[" + os.getenv("SERVER_NAME") +
                "] - (gunicorn) pid:(%(process)d) thread:(%(thread)d) - %(message)s"
            ),
            "datefmt": "%d/%m/%Y %H:%M:%S"
        },
        'generic': {
            'format': (
                "[%(asctime)s]:[%(levelname)s]:[" + os.getenv("SERVER_NAME") +
                "] - (gunicorn) pid:(%(process)d) thread:(%(thread)d) - %(message)s"
            ),
            'datefmt': "%d/%m/%Y %H:%M:%S",
            '()': 'logging.Formatter',
        },
        'app': {
            'format': (
                "[%(asctime)s]:[%(levelname)s]:[" + os.getenv("SERVER_NAME") +
                "] - (server)   pid:(%(process)d) thread:(%(thread)d) - %(message)s"
            ),
            'datefmt': "%d/%m/%Y %H:%M:%S",
            '()': 'logging.Formatter',
        }
    },
    'handlers': {
        'error_log': {
            'class': 'logging.StreamHandler',
            'formatter': 'generic'
        },
        'access_log': {
            'class': 'logging.StreamHandler',
            'formatter': 'access'
        },
        'app_log': {
            'class': 'logging.StreamHandler',
            'formatter': 'app'
        },
    },
    'loggers': {
        'gunicorn.error': {
            'level': os.getenv("LOG_LEVEL"),
            'handlers': ['error_log'],
            'propagate': False,
        },
        'gunicorn.access': {
            'level': os.getenv("LOG_LEVEL"),
            'handlers': ['access_log'],
            'propagate': True,
        },
        "app.log": {
            'level': os.getenv("LOG_LEVEL"),
            'handlers': ['app_log'],
            'propagate': False,
        }
    },
}
