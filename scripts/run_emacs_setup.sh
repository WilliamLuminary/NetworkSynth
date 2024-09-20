#!/bin/bash

# Define blue color (similar to Apple's terminal blue)
BLUE='\033[38;5;33m'
RESET='\033[0m'  # Reset color to default

# Run Emacs in batch mode to install packages and exit
echo -e "${BLUE}Running Emacs in the background to install packages...${RESET}"
emacs --batch -l ~/.emacs.d/init.el --eval="(progn (package-refresh-contents) (package-install-selected-packages))"

echo -e "${BLUE}Emacs setup and package installation complete!${RESET}"