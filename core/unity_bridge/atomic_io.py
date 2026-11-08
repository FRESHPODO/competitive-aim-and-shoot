"""Retry transient Windows reader/antivirus locks on atomic file replacement."""
import os
import time


def replace_with_retry(source, destination, *, attempts=20, delay=.05):
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt+1 == attempts:
                raise
            time.sleep(delay)
