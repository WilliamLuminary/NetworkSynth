# src/utils/debug_utils.py

import functools
import time

from rich import print as rich_print

DEBUG = False


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            rich_print(f"[DEBUG] Time taken by \"{func.__name__}\": {round(end - start, 2)} seconds")
            return result
        else:
            return func(*args, **kwargs)

    return wrapper

