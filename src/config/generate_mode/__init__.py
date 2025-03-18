# src/config/generate_mode/__init__.py

# Sample configuration is named as ".*Config$"
# Others are named as ".*Config\d+"
from .config_1 import Config1 as GenConfig1
from .config_2 import Config2 as GenConfig2
from .sample_pos_and_mat import SampleConfig as GenConfig
from .sample_igraph import SampleConfig as IgraphGenConfig
