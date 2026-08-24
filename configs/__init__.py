from . import (
    compare_mode,
    generate_mode,
    hybrid_mode,
    mosaic_mode,
    scaling_mode,
    sweep_mode,
)
from .base_config import BaseConfig  # noqa: F401
from .enums import (  # noqa: F401
    AnalysisMode,
    DatasetId,
)
from .file_definitions import (  # noqa: F401
    DEFAULT_SAVE_SPECS,
    FILE_CONFIGURATIONS,
    FileConfig,
    ImageConfig,
    PlotConfig,
    save_csv,
    save_network_csv,
    save_network_nkbin,
    save_networkit,
    save_pickle,
    save_png,
    save_svg,
    save_webp,
)
from .params import SynthParams  # noqa: F401

#: Which mode package owns each config name.  Built from the mode indexes,
#: which are filename-derived, so no config module is imported to build it.
_CONFIG_MODULES = {
    _name: _module
    for _module in (
        compare_mode,
        generate_mode,
        hybrid_mode,
        mosaic_mode,
        scaling_mode,
        sweep_mode,
    )
    for _name in _module.__all__
}


def __getattr__(name):
    """Resolve ``configs.GenConfig`` and friends on first use.

    A run needs one config; importing the other sixteen to put their names in
    this namespace ran their module-level path arithmetic and pulled in cv2 for
    nothing.  The name is looked up here and the module imported only then.
    """
    module = _CONFIG_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    config_class = getattr(module, name)
    globals()[name] = config_class
    return config_class


def __dir__():
    return sorted(set(_CONFIG_MODULES) | set(globals()))
