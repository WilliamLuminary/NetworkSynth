#!/bin/bash

BLUE='\033[38;5;33m'
RED='\033[38;5;196m'
RESET='\033[0m'

echo -e "${BLUE}Updating package list and installing Emacs...${RESET}"

if sudo apt update && sudo apt install -y emacs; then
  echo -e "${BLUE}Installation complete!${RESET}"
else
  echo -e "${RED}Failed to update package list or install Emacs. Please check your system setup and try again.${RESET}"
  exit 1
fi
