"""RSS logging helper for router daemon (task 3B.1)."""

from __future__ import annotations

import logging
import resource
import sys

logger = logging.getLogger(__name__)


def log_rss_mb(stage: str) -> float:
    """Log resident set size in MB; returns measured value."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        mb = usage / (1024 * 1024)
    else:
        mb = usage / 1024
    logger.info("RSS [%s]: %.1f MB", stage, mb)
    return mb
