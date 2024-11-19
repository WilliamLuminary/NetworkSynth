#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Starting full setup for Git and Emacs deployment on Colab...${RESET}"

"$SCRIPT_DIR/setup_install.sh" || { echo "Failed to update and install"; exit 1; }
"$SCRIPT_DIR/setup_git.sh" || { echo "Git setup failed"; exit 1; }
"$SCRIPT_DIR/setup_emacs.sh" || { echo "Emacs config setup failed"; exit 1; }

echo -e "${BLUE}Full setup complete!${RESET}"