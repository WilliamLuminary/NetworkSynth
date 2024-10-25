#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Setting up Git...${RESET}"

git config --global core.editor "emacs"
git config --global core.fileMode false

git config --global user.name "Yaxing Li"
git config --global user.email "93109493+WilliamLuminary@users.noreply.github.com"

echo -e "${BLUE}Git configuration complete!${RESET}"
