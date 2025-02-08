#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

BLUE='\033[38;5;33m'
RED='\033[38;5;196m'
RESET='\033[0m'

#----------------- Github token -----------------#
TOKEN_FILE="$SCRIPT_DIR/../credentials/github_token"

echo -e "${BLUE}Setting up GitHub credentials...${RESET}"

if [ ! -f "$TOKEN_FILE" ]; then
  echo -e "${RED}Token file '$TOKEN_FILE' not found. Please create the file with your GitHub token.${RESET}"
  return 1
fi

GITHUB_TOKEN=$(cat "$TOKEN_FILE")
export GITHUB_TOKEN

REPO_URL="https://api.github.com/user/repos"
echo -e "${BLUE}Validating GitHub token...${RESET}"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $GITHUB_TOKEN" "$REPO_URL")

if [ "$RESPONSE" -ne 200 ]; then
  echo -e "${RED}Invalid GitHub token. Please check the token and try again.${RESET}"
  return 1
fi

echo -e "${BLUE}GitHub token is valid. Credentials setup complete!${RESET}"
return 0