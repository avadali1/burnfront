"""Process pools that do not re-run the user's script in the workers.

With the 'spawn' start method (always on Windows and macOS), each worker first re-imports the parent's
``__main__`` script. A script without an ``if __name__ == '__main__':`` guard then runs again inside
every worker, which starts its own pool, and the pool breaks. Burnfront's workers only need functions
from the burnfront package itself, so the parent's ``__main__`` is hidden from multiprocessing while the
workers are started.
"""
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager


@contextmanager
def _main_hidden():
    main = sys.modules.get('__main__')
    if main is None:
        yield
        return
    missing = object()
    saved = {k: getattr(main, k, missing) for k in ('__file__', '__spec__')}
    try:
        if saved['__file__'] is not missing:
            del main.__file__
        main.__spec__ = None
        yield
    finally:
        for k, v in saved.items():
            if v is missing:
                if hasattr(main, k):
                    delattr(main, k)
            else:
                setattr(main, k, v)


@contextmanager
def spawn_pool(n_jobs):
    """ProcessPoolExecutor (spawn) with the parent's ``__main__`` hidden while the pool exists.

    Every task must be a function importable from burnfront, with picklable arguments."""
    with _main_hidden():
        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context('spawn')) as ex:
            yield ex
