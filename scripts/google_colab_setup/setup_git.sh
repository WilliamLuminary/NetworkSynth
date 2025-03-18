#!/bin/bash

BLUE='\033[38;5;33m'
RED='\033[38;5;196m'
RESET='\033[0m'

echo -e "${BLUE}Setting up local Git configurations...${RESET}"
git config user.name "Yaxing Li (Google Colab)"
git config user.email "93109493+WilliamLuminary@users.noreply.github.com"
#git config core.editor "emacs"
git config core.fileMode false
echo -e "${BLUE}Local Git configuration complete!${RESET}"


if [ -z "$GITHUB_TOKEN" ]; then
  echo -e "${RED}GITHUB_TOKEN is not set. Please run setup_credentials.sh first.${RESET}"
  return 1
fi

echo -e "${BLUE}GitHub token is valid. Proceeding to set remote URL...${RESET}"

git remote set-url origin https://"$GITHUB_TOKEN"@github.com/WilliamLuminary/NetworkSynth.git || {
  echo -e "${RED}Failed to set remote URL. Please check your repository and token.${RESET}"
  return 1
}

echo -e "${BLUE}GitHub remote URL set successfully!${RESET}"
return 0