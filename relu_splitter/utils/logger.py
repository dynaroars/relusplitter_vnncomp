import sys
import logging

# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
#     stream=sys.stdout,
#     level=logging.DEBUG
# )

default_logger = logging.getLogger(__name__)
default_logger.formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
default_logger_handler = logging.StreamHandler(sys.stdout)
default_logger.setLevel(logging.DEBUG)

logger=default_logger