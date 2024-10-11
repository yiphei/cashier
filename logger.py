# my_logger.py
import logging
import sys

# Create a custom logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

# Create formatters for different log levels
formatter_debug = logging.Formatter('%(asctime)s - DEBUG - %(message)s')

# Handlers with different formatters
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter_debug)

# Add handlers to the logger
logger.addHandler(debug_handler)