import logging
import shutil

LOGGER = logging.getLogger(__name__)


def log_disk_usage():
    BytesPerGB = 1024 * 1024 * 1024

    (total, used, free) = shutil.disk_usage(".")
    LOGGER.info("Disk Usage - total: %.2fGB" % (float(total)/BytesPerGB) + ", Used:  %.2fGB" % (float(used)/BytesPerGB))