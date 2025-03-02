# src/config/__init__.py

from .base_config import BaseConfig
from .config.config_1 import Config1
from .config.config_2 import Config2
from .config.multifractal_analysis_config import MultifractalConfig
from .sample.generate_mode import GenerateModeConfigSample
from .sample.attributes_generate_mode import AttributesGenerateModeConfigSample

from .enums import DataType, FileTag, Resolution, SetName, AnalysisMode
from .file_definitions import FileConfig, ImageConfig, PlotConfig, FILE_CONFIGURATIONS
