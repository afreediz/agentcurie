import logging
import sys
import json
import os
from logging.handlers import RotatingFileHandler

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError(f'{levelName} already defined in logging module')
    if hasattr(logging, methodName):
        raise AttributeError(f'{methodName} already defined in logging module')
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError(f'{methodName} already defined in logger class')

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def setup_logging(stream=None, log_level=None, force_setup=False):
    """
    Setup logging configuration for browser-use.
    """
    try:
        addLoggingLevel('RESULT', 35)
    except AttributeError:
        pass

    log_type = log_level

    if logging.getLogger().hasHandlers() and not force_setup:
        return logging.getLogger('gpa')

    root = logging.getLogger()
    root.handlers = []

    class BrowserUseFormatter(logging.Formatter):
        def format(self, record):
            return super().format(record)

    console = logging.StreamHandler(stream or sys.stdout)

    if log_type == 'result':
        console.setLevel('RESULT')
        console.setFormatter(BrowserUseFormatter('%(message)s'))
    else:
        console.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

    root.addHandler(console)

    if log_type == 'result':
        root.setLevel('RESULT')
    elif log_type == 'debug':
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    gpa_logger = logging.getLogger('gpa')
    gpa_logger.propagate = False
    gpa_logger.addHandler(console)
    gpa_logger.setLevel(root.level)

    logger = logging.getLogger('gpa')

    third_party_loggers = [
        'WDM',
        'httpx',
        'selenium',
        'playwright',
        'urllib3',
        'asyncio',
        'langsmith',
        'langsmith.client',
        'openai',
        'httpcore',
        'charset_normalizer',
        'anthropic._base_client',
        'PIL.PngImagePlugin',
        'trafilatura.htmlprocessing',
        'trafilatura',
        'groq',
        'portalocker',
        'portalocker.utils',
    ]

    for logger_name in third_party_loggers:
        third_party = logging.getLogger(logger_name)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "gpa.log")

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)

    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger