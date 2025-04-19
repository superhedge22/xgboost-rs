#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XGB_DIR="$SCRIPT_DIR/xgboost"

echo "Setting up XGBoost in $XGB_DIR"

# Check if directory already exists
if [ -d "$XGB_DIR" ]; then
    echo "XGBoost directory already exists."
    
    # Verify it's a valid git repository
    if [ -d "$XGB_DIR/.git" ]; then
        echo "XGBoost appears to be a valid git repository."
        
        # Update to the correct branch
        cd "$XGB_DIR"
        git fetch -all 2>/dev/null || true
        git checkout release_3.0.0 2>/dev/null || true
        
        # Initialize submodules if needed
        git submodule update --init --recursive
        
        echo "XGBoost repository is now up-to-date."
        exit 0
    else
        echo "XGBoost directory exists but doesn't seem to be a valid git repository."
        echo "Removing existing directory to set up a fresh clone..."
        rm -rf "$XGB_DIR"
    fi
fi

# Directory doesn't exist or was invalid, clone it
echo "Cloning XGBoost repository..."
git clone --branch release_3.0.0 --depth 1 https://github.com/dmlc/xgboost "$XGB_DIR"

# Initialize XGBoost's submodules
cd "$XGB_DIR"
git submodule update --init --recursive

echo "XGBoost setup complete." 