import logging
import colorlog

class LogHandler(logging.Logger):

    def __init__(self, level):
        super(LogHandler, self).__init__(name="SelfDefinedLogger")
        self.config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        self.formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s %(filename)s %(levelname)s %(module)s.py: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=self.config)

        self.setLevel(level=level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)

        self.addHandler(console_handler)


logger = LogHandler(level=logging.INFO)

__all__ = [
    "logger"
]
