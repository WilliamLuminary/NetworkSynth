#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Running Emacs in the background to install packages...${RESET}"
emacs --batch -l ~/.emacs.d/init.el --eval="(progn (package-refresh-contents) (package-install-selected-packages))"

echo -e "${BLUE}Emacs setup and package installation complete!${RESET}"