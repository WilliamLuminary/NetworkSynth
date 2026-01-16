# src/config/generate_mode/__init__.py

# Sample configuration is named as ".*Config$"
# Others are named as ".*Config\d+"
from .config_1 import Config1 as GenConfig1
from .config_2 import Config2 as GenConfig2
from .config_nanowires import NanowiresConfig as GenConfigNanowires
from .config_sample import SampleConfig as GenConfig
