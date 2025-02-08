#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

BLUE='\033[38;5;33m'
RED='\033[38;5;196m'
RESET='\033[0m'

echo -e "${BLUE}Starting full setup Google Colab Env...${RESET}"

#----------------- Set up credential files -----------------#
if ! source "$SCRIPT_DIR/setup_credentials.sh"; then
  echo -e "${RED}Credential setup failed. Aborting.${RESET}"
  exit 1
fi

#----------------- Set up git -----------------#
if ! "$SCRIPT_DIR/setup_git.sh"; then
  echo -e "${RED}Git setup failed. Aborting.${RESET}"
  exit 1
fi

#----------------- Set up packages -----------------#
if ! "$SCRIPT_DIR/setup_install.sh"; then
  echo -e "${RED}Failed to update and install required packages. Aborting.${RESET}"
  exit 1
fi

#----------------- Set up emacs -----------------#
echo -e "${BLUE}Running Emacs config setup in the background...${RESET}"
"$SCRIPT_DIR/setup_emacs.sh" &
EMACS_PID=$!

wait $EMACS_PID
EMACS_EXIT_CODE=$?

if [ $EMACS_EXIT_CODE -ne 0 ]; then
  echo -e "${RED}Emacs config setup failed with exit code $EMACS_EXIT_CODE. Aborting.${RESET}"
  exit 1
fi

echo -e "${BLUE}Full setup complete!${RESET}"
