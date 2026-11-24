"""Sea surface height forecasting — INCOIS prototype."""
import logging
import os

_level = os.getenv("SSH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
