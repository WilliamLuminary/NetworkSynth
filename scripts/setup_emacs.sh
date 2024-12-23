#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

BLUE='\033[38;5;33m'
RESET='\033[0m'

if [ -z "$TOKEN" ]; then
    echo "Error: TOKEN is not set. Please run setup_git.sh first."
    exit 1
fi

EMACS_REPO_URL="https://$TOKEN@github.com/WilliamLuminary/emacs-gui-config.git"
CLONE_DIR="/content/emacs-config"

echo -e "${BLUE}Cloning Emacs configuration repository...${RESET}"
git clone $EMACS_REPO_URL $CLONE_DIR || { echo "Failed to clone Emacs configuration repository"; exit 1; }

echo -e "${BLUE}Setting up Emacs configuration...${RESET}"
mkdir -p ~/.emacs.d
cp -r $CLONE_DIR/* ~/.emacs.d/ || { echo "Failed to copy configuration files"; exit 1; }

echo -e "${BLUE}Running Emacs setup to install packages...${RESET}"
bash "$SCRIPT_DIR/setup_run_emacs.sh" || { echo "Emacs package installation failed"; exit 1; }

echo -e "${BLUE}Emacs setup complete!${RESET}"