#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Setting up Emacs configuration...${RESET}"
mkdir -p ~/.emacs.d
cp -r /content/emacs-config/* ~/.emacs.d/ || { echo "Failed to copy configuration files"; exit 1; }

echo -e "${BLUE}Running Emacs setup to install packages...${RESET}"
bash ./scripts/run_emacs_setup.sh || { echo "Emacs package installation failed"; exit 1; }

echo -e "${BLUE}Emacs setup complete!${RESET}"