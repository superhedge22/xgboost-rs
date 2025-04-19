# XGBoost Rust Bindings

Rust bindings for the [XGBoost](https://github.com/dmlc/xgboost) gradient boosting library.

## Installation

This crate provides Rust bindings for the XGBoost C API.

Building this crate will automatically build XGBoost from source and initialize the git submodules if needed. The following dependencies are required:

- A C++ compiler (g++, clang, or MSVC)
- CMake (3.14 or higher)
- OpenMP (optional but recommended)
- Git (for automatic submodule initialization)

> Note: If the automatic initialization fails, you may need to run `git submodule update --init --recursive` manually.

Currently supported version: [v3.0.0](https://github.com/dmlc/xgboost/releases/tag/v3.0.0)

## Troubleshooting

### Submodule initialization issues

If you encounter errors like:
```
XGBoost directory still not found after submodule initialization. Please check your git configuration.
```

You can try:

1. Run the included setup script:
   ```bash
   ./setup_xgboost.sh
   ```

2. Or manually clone the XGBoost repository:
   ```bash
   git clone --branch release_3.0.0 --depth 1 https://github.com/dmlc/xgboost xgboost-rs-sys/xgboost
   cd xgboost-rs-sys/xgboost
   git submodule update --init --recursive
   ```

3. Check that your `.gitmodules` file contains the correct URL and branch:
   ```
   [submodule "xgboost-rs-sys/xgboost"]
       path = xgboost-rs-sys/xgboost
       url = https://github.com/dmlc/xgboost
       branch = release_3.0.0
   ```

## Important Note for macOS Users

With the upgrade to XGBoost 3.0.0, **OpenMP is now required** for proper operation on macOS. **Xcode** is also required for development on macOS.

Quick setup for macOS:
```bash
# Check if brew is installed and install OpenMP
if command -v brew &> /dev/null; then
    brew install libomp
else
    echo "Homebrew not found. Please install it from https://brew.sh/ and then run 'brew install libomp'"
    # Or you can use the script in xgboost-rs-sys
    # cd xgboost-rs-sys && ./install_deps.sh
fi

# When running tests, you may need to set these environment variables
export LIBRARY_PATH="$(brew --prefix libomp)/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

See the [xgboost-rs-sys README](xgboost-rs-sys/README.md) for more detailed instructions.

## Dependencies
- See [Build XGBoost from Source](https://xgboost.readthedocs.io/en/stable/build.html#building-from-source)
- CMake 3.14 or higher
- A recent C++ compiler supporting C++11 (g++-5.0 or higher)
- libclang
- OpenMP library (required for macOS with XGBoost 3.0.0)
- Xcode (required for macOS)

## Supported Platforms

This library has been tested and should work on the following platforms:
- macOS with Intel processors
- macOS with Apple Silicon (M1/M2/M3)
- Linux AMD64 (x86_64)
- Linux ARM64

For platform-specific build instructions, see the [xgboost-rs-sys README](xgboost-rs-sys/README.md).
