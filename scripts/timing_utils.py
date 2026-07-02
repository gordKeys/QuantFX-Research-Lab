from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(label, results=None):
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    elapsed_days = elapsed / 86400
    if results is not None:
        results[label] = elapsed_days
    print(f"{label}: {elapsed_days:.6f} days")
