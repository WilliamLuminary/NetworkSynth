#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Cloning Emacs config repository...${RESET}"
git clone https://[Your token]@github.com/WilliamLuminary/emacs-gui-config.git /content/emacs-config

echo -e "${BLUE}Repository cloned successfully!${RESET}"