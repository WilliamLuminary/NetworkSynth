#!/bin/bash

BLUE='\033[38;5;33m'
RESET='\033[0m'

echo -e "${BLUE}Updating and installing tree and emacs...${RESET}"
#sudo apt update && sudo apt install -y tree emacs
sudo apt install -y emacs
echo -e "${BLUE}Installation complete!${RESET}"
