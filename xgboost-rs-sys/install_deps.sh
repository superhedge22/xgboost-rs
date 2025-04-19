#!/bin/bash
set -e

# Detect OS
OS=$(uname -s)

echo "Installing dependencies for xgboost-rs-sys on $OS"

if [[ "$OS" == "Darwin" ]]; then
    # macOS - we need to install libomp
    echo "macOS detected. Installing OpenMP library (required for XGBoost)..."
    
    if command -v brew >/dev/null 2>&1; then
        echo "Homebrew found. Installing libomp..."
        brew install libomp
        
        echo "Setting environment variables for OpenMP..."
        LIBOMP_PREFIX=$(brew --prefix libomp)
        
        # Print instructions for the user
        echo ""
        echo "================ IMPORTANT ================"
        echo "OpenMP installed successfully at: $LIBOMP_PREFIX"
        echo ""
        echo "When running tests or applications that use xgboost-rs,"
        echo "you may need to set these environment variables:"
        echo ""
        echo "export LIBRARY_PATH=\"$LIBOMP_PREFIX/lib:\$LIBRARY_PATH\""
        echo "export DYLD_LIBRARY_PATH=\"$LIBOMP_PREFIX/lib:\$DYLD_LIBRARY_PATH\""
        echo "================ IMPORTANT ================"
        echo ""
    else
        echo "Error: Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "Then run this script again."
        exit 1
    fi
elif [[ "$OS" == "Linux" ]]; then
    # Linux - install OpenMP development packages
    echo "Linux detected. Installing OpenMP development packages..."
    
    if command -v apt-get >/dev/null 2>&1; then
        # Debian/Ubuntu
        echo "Debian/Ubuntu system detected"
        sudo apt-get update
        sudo apt-get install -y libgomp1
    elif command -v dnf >/dev/null 2>&1; then
        # Fedora/RHEL
        echo "Fedora/RHEL system detected"
        sudo dnf install -y libgomp
    elif command -v yum >/dev/null 2>&1; then
        # Older RHEL/CentOS
        echo "CentOS/RHEL system detected"
        sudo yum install -y libgomp
    else
        echo "Unsupported Linux distribution. Please install OpenMP development packages manually."
        exit 1
    fi
else
    echo "Unsupported operating system: $OS"
    echo "Please install OpenMP development packages manually."
    exit 1
fi

echo "Dependencies installed successfully!" 