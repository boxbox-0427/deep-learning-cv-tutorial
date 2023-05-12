import logging


logging.basicConfig(format='%(asctime)s-%(levelname)s-%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

__all__ = [
    "logger"
]