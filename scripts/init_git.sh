#!/bin/bash

# Define blue color (similar to Apple's terminal blue)
BLUE='\033[38;5;33m'
RESET='\033[0m'  # Reset color to default

# Update and install tree and emacs
echo -e "${BLUE}Updating and installing tree and emacs...${RESET}"
sudo apt update && sudo apt install -y tree emacs

# Setup Git
echo -e "${BLUE}Setting up Git...${RESET}"
git config --global user.name "Yaxing Li"
git config --global user.email "93109493+WilliamLuminary@users.noreply.github.com"

# Clone your Emacs configuration repo using a personal access token
echo -e "${BLUE}Cloning Emacs config repository...${RESET}"
git clone https://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@github.com/WilliamLuminary/emacs-gui-config.git /content/emacs-config

# Copy the config files to your Emacs directory
echo -e "${BLUE}Setting up Emacs configuration...${RESET}"
mkdir -p ~/.emacs.d
cp -r /content/emacs-config/* ~/.emacs.d/ || { echo "Failed to copy configuration files"; exit 1; }

# Run the Emacs setup script to install packages
echo -e "${BLUE}Running Emacs setup...${RESET}"
bash ./scripts/run_emacs_setup.sh

# Done
echo -e "${BLUE}Setup complete!${RESET}"