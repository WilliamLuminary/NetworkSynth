#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Validating GitHub token...${RESET}"

TOKEN=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
REPO_URL="https://api.github.com/user/repos"

RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $TOKEN" $REPO_URL)

if [ "$RESPONSE" -ne 200 ]; then
    echo -e "${BLUE}Invalid GitHub token. Please check the token and try again.${RESET}"
    exit 1
fi

echo -e "${BLUE}GitHub token is valid. Proceeding...${RESET}"

echo -e "${BLUE}Cloning Emacs config repository...${RESET}"

git clone https://$TOKEN@github.com/WilliamLuminary/emacs-gui-config.git /content/emacs-config && \
git remote set-url origin https://$TOKEN@github.com/WilliamLuminary/NetworkSynth.git

if [ $? -eq 0 ]; then
    echo -e "${BLUE}Repository cloned successfully!${RESET}"
else
    echo -e "${BLUE}Failed to clone repository. Please check the token and repository URL.${RESET}"
    exit 1
fi