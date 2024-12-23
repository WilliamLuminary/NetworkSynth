#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Setting up global Git configurations...${RESET}"
git config --global user.name "Yaxing Li"
git config --global user.email "93109493+WilliamLuminary@users.noreply.github.com"
git config --global core.editor "emacs"
git config --global core.fileMode false
echo -e "${BLUE}Global Git configuration complete!${RESET}"

echo -e "${BLUE}Setting up local Git configurations...${RESET}"
git config user.name "Yaxing Li (Google Colab)"
git config user.email "93109493+WilliamLuminary@users.noreply.github.com"
git config core.fileMode false
echo -e "${BLUE}Local Git configuration complete!${RESET}"

TOKEN=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
REPO_URL="https://api.github.com/user/repos"

echo -e "${BLUE}Validating GitHub token...${RESET}"

RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $TOKEN" $REPO_URL)

if [ "$RESPONSE" -ne 200 ]; then
  echo -e "${BLUE}Invalid GitHub token. Please check the token and try again.${RESET}"
  exit 1
fi

echo -e "${BLUE}GitHub token is valid. Proceeding to set remote URL...${RESET}"

git remote set-url origin https://$TOKEN@github.com/WilliamLuminary/NetworkSynth.git
echo -e "${BLUE}GitHub remote URL set successfully!${RESET}"
