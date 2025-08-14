import logging
import sys
from importlib.metadata import PackageNotFoundError, version

# Create a custom logger
curricula_logger = logging.getLogger(__name__)
curricula_logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)

stdout_handler.setFormatter(
    logging.Formatter("%(message)s")
)  # This makes the logs easier to parse from the client side
stderr_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
curricula_logger.addHandler(stdout_handler)
curricula_logger.addHandler(stderr_handler)


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
