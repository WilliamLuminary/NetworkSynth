#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

BLUE='\033[38;5;33m'
RED='\033[38;5;196m'
RESET='\033[0m'

if [ -z "$GITHUB_TOKEN" ]; then
  echo -e "${RED}Error: TOKEN is not set. Please run setup_git.sh first.${RESET}"
  exit 1
fi

EMACS_REPO_URL="https://$GITHUB_TOKEN@github.com/WilliamLuminary/emacs-gui-config.git"
CLONE_DIR="/content/emacs-config"

echo -e "${BLUE}Cloning Emacs configuration repository...${RESET}"
git clone "$EMACS_REPO_URL" $CLONE_DIR || {
  echo -e "${RED}Failed to clone Emacs configuration repository.${RESET}"
  return 1
}

echo -e "${BLUE}Setting up Emacs configuration...${RESET}"
mkdir -p ~/.emacs.d
cp -r $CLONE_DIR/* ~/.emacs.d/ || {
  echo -e "${RED}Failed to copy configuration files.${RESET}"
  return 1
}

echo -e "${BLUE}Running Emacs setup to install packages...${RESET}"
bash "$SCRIPT_DIR/setup_run_emacs.sh" || {
  echo -e "${RED}Emacs package installation failed.${RESET}"
  return 1
}

echo -e "${BLUE}Emacs setup complete!${RESET}"
return 0