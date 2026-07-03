#!/bin/bash

# Setup script for AI Red Teamer project
# This script installs all required packages including the HTB AI Library

set -e  # Exit on error

ENV_NAME="ai_red_teamer"
REQUIREMENTS_FILE="requirements.txt"
GIT_PACKAGE="git+https://github.com/PandaSt0rm/htb-ai-library"

echo "Setting up AI Red Teamer environment..."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install conda/miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Found existing conda environment: ${ENV_NAME}"
else
    echo "Warning: Environment '${ENV_NAME}' not found."
    read -p "Would you like to create it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating conda environment: ${ENV_NAME}"
        conda create -n ${ENV_NAME} python=3.11 -y
    else
        echo "Exiting. Please create the environment manually or choose an existing one."
        exit 1
    fi
fi

# Activate environment
echo "Activating conda environment: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Check if requirements.txt exists
if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "Error: ${REQUIREMENTS_FILE} not found in current directory"
    exit 1
fi

# Install packages from requirements.txt (excluding git package)
echo ""
echo "Installing packages from ${REQUIREMENTS_FILE}..."
pip install -r <(grep -v "git+https" ${REQUIREMENTS_FILE}) || {
    echo "Warning: Some packages from requirements.txt may have failed, but continuing..."
}

# Install git package separately
echo ""
echo "Installing HTB AI Library from GitHub..."
pip install --upgrade ${GIT_PACKAGE} || {
    echo "Warning: Failed to install HTB AI Library from git"
    echo "   You may need to install it manually: pip install --upgrade ${GIT_PACKAGE}"
}

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
