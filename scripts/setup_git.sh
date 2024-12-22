#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Setting up global Git configurations...${RESET}"
git config --global user.name "Yaxing Li"
git config --global user.email "93109493+WilliamLuminary@users.noreply.github.com"
git config --global core.editor "emacs"
git config --global core.fileMode false
echo -e "${BLUE}Setting up local Git configurations...${RESET}"

git config user.name "Yaxing Li (Google Colab)"
git config user.email "93109493+WilliamLuminary@users.noreply.github.com"
git config core.fileMode false
echo -e "${BLUE}Git configuration (global and local) complete!${RESET}"