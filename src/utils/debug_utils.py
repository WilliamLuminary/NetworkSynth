# src/utils/debug_utils.py

import builtins
import functools
import time
from rich import print as rich_print

DEBUG = False  # Set to True to enable debug prints


def debugging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_print = builtins.print

        def debug_print(*print_args, debug=True, **print_kwargs):
            if DEBUG and debug:
                rich_print(f"[DEBUG] \"{func.__name__}\":", *print_args, **print_kwargs)
            elif not debug:
                original_print(*print_args, **print_kwargs)

        builtins.print = debug_print
        try:
            return func(*args, **kwargs)
        finally:
            builtins.print = original_print

    return wrapper


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
