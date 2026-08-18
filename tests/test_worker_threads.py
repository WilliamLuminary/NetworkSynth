# tests/test_worker_threads.py
"""Pool workers must pin networkit to a single thread.

networkit defaults to one thread per core. A worker forked from a parent that
has already run networkit inherits an OpenMP runtime whose threads do not exist
in the child; the child then spins instead of working. On a 128-core host that
turned a three-second generate run into one that never finished, and it is
silent — no error, no warning, just a process burning CPU forever.

`pipelines/hybrid.py` pinned its tile worker long ago. `generate`, `mosaic` and
`sweep` did not, and these tests exist so that gap cannot reopen. Each worker
pins before its first early return, so a worker called with an already-set exit
event still proves the pin ran.
"""

import threading

import networkit as nk
import pytest

from configs.params import SynthParams

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_thread_count():
    """The pin is global to networkit, so put it back for other tests."""
    original = nk.getMaxNumberOfThreads()
    yield
    nk.setNumberOfThreads(original)


@pytest.fixture
def stopped_event():
    event = threading.Event()
    event.set()
    return event


def _params():
    return SynthParams(
        synthetic_frame_size=(64, 64),
        closed_nodes_factor=1.2,
        closed_edges_factor=0.8,
        max_attempts=1,
        seed=1,
    )


def test_generate_worker_pins_networkit(restore_thread_count, stopped_event):
    from pipelines.generate import generate_synthetic_network

    nk.setNumberOfThreads(4)

    generate_synthetic_network(stopped_event, None, None, None, _params())

    assert nk.getMaxNumberOfThreads() == 1


def test_sweep_worker_pins_networkit(restore_thread_count, stopped_event):
    from pipelines.sweep import _generate_with_factors

    nk.setNumberOfThreads(4)

    _generate_with_factors(stopped_event, None, None, None, _params())

    assert nk.getMaxNumberOfThreads() == 1


def test_mosaic_worker_pins_networkit(restore_thread_count, stopped_event):
    from pipelines.mosaic import generate_single_tile

    nk.setNumberOfThreads(4)

    generate_single_tile((0, 0, stopped_event, None, (64, 64), 0.0, 0.0, _params()))

    assert nk.getMaxNumberOfThreads() == 1


def test_generate_select_workers_pin_networkit(restore_thread_count, stopped_event):
    """generate_select has its own pool over two workers defined in generate.py."""
    from pipelines.generate import (
        _generate_single_network,
        _generate_single_network_collecting_snapshots,
    )

    nk.setNumberOfThreads(4)
    _generate_single_network(stopped_event, None, None, None, _params())
    assert nk.getMaxNumberOfThreads() == 1

    nk.setNumberOfThreads(4)
    _generate_single_network_collecting_snapshots(
        stopped_event, None, None, None, 0, 0, _params()
    )
    assert nk.getMaxNumberOfThreads() == 1


def test_generate_from_props_worker_pins_networkit(restore_thread_count, stopped_event):
    """It defines its own worker rather than reusing generate's."""
    from pipelines.generate_from_props import generate_synthetic_network

    nk.setNumberOfThreads(4)

    generate_synthetic_network(stopped_event, None, _params())

    assert nk.getMaxNumberOfThreads() == 1


def test_every_pool_worker_is_covered_here():
    """A pool worker that pins nothing is the bug that hung a default run.

    Checks the source of every function submitted to a ProcessPoolExecutor,
    excluding the plot pools, whose workers render images and never touch
    networkit.
    """
    import inspect

    import pipelines.generate as gen
    import pipelines.generate_from_props as props
    import pipelines.hybrid as hyb
    import pipelines.mosaic as mos
    import pipelines.sweep as swp

    workers = [
        gen.generate_synthetic_network,
        gen._generate_single_network,
        gen._generate_single_network_collecting_snapshots,
        props.generate_synthetic_network,
        hyb._generate_tile_worker,
        mos.generate_single_tile,
        swp._generate_with_factors,
    ]
    unpinned = [
        w.__qualname__
        for w in workers
        if "setNumberOfThreads(1)" not in inspect.getsource(w)
    ]
    assert not unpinned, f"pool workers not pinning networkit: {unpinned}"


def test_hybrid_tile_worker_still_pins_networkit(restore_thread_count):
    """Hybrid was already correct; this guards against it being undone."""
    import inspect

    from pipelines.hybrid import _generate_tile_worker

    assert "setNumberOfThreads(1)" in inspect.getsource(_generate_tile_worker)
