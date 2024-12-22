#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Cloning Emacs config repository...${RESET}"

TOKEN=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
git clone https://$TOKEN@github.com/WilliamLuminary/emacs-gui-config.git /content/emacs-config

echo -e "${BLUE}Repository cloned successfully!${RESET}"