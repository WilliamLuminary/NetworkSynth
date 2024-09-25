#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Cloning Emacs config repository...${RESET}"
git clone https://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@github.com/WilliamLuminary/emacs-gui-config.git /content/emacs-config

echo -e "${BLUE}Repository cloned successfully!${RESET}"