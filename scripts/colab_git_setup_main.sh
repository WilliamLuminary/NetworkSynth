#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Starting full setup for Git and Emacs deployment on Colab...${RESET}"

"$SCRIPT_DIR/colab_git_setup_install.sh" || { echo "Failed to update and install"; exit 1; }
"$SCRIPT_DIR/colab_git_setup_git.sh" || { echo "Git setup failed"; exit 1; }
"$SCRIPT_DIR/colab_git_setup_clone.sh" || { echo "Failed to clone Emacs config repository"; exit 1; }
"$SCRIPT_DIR/colab_git_setup_emacs.sh" || { echo "Emacs config setup failed"; exit 1; }

echo -e "${BLUE}Full setup complete!${RESET}"