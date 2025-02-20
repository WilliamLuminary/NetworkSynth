# src/config/__init__.py

from .base_config import BaseConfig
from .config import Config1, Config2, MultifractalConfig
from .sample import GenerateModeConfigSample, AttributesGenerateModeConfigSample

from .enums import DataType, FileTag, Resolution, SetName, AnalysisMode
from .file_definitions import FileConfig, ImageConfig, PlotConfig, FILE_CONFIGURATIONS
