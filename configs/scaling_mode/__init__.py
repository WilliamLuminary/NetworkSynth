from pathlib import Path

from .._loader import index_configs, load_config

_INDEX = index_configs(Path(__file__).parent, mode_prefix="Scaling")
__all__ = sorted(_INDEX)


def __getattr__(name):
    """Import a config only when it is asked for.

    The names are known from the filenames, so ``dir()`` and ``__all__`` are
    complete without importing anything; the module behind a name is loaded on
    first access and cached in this namespace.
    """
    module_stem = _INDEX.get(name)
    if module_stem is None:
        raise AttributeError(f"module {__name__!r} has no config {name!r}")
    config_class = load_config(__name__, module_stem, name)
    globals()[name] = config_class
    return config_class


def __dir__():
    return sorted(set(__all__) | set(globals()))
