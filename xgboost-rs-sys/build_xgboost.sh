#!/bin/bash
set -e

# Go to the xgboost directory
cd "$(dirname "$0")/xgboost"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

echo "Building for OS: $OS, Architecture: $ARCH"

# Set build options based on platform
CMAKE_OPTIONS="-DBUILD_STATIC_LIB=ON -DBUILD_SHARED_LIBS=OFF"

# Handle OpenMP settings
if [[ "$OS" == "Darwin" ]]; then
    # macOS - OpenMP requires special handling
    if [[ "${USE_OPENMP:-1}" == "1" ]]; then
        echo "Enabling OpenMP support on macOS"
        
        # Try to find Homebrew-installed libomp
        if command -v brew >/dev/null 2>&1; then
            LIBOMP_PREFIX=$(brew --prefix libomp 2>/dev/null || echo "")
            if [[ -n "$LIBOMP_PREFIX" ]]; then
                echo "Found Homebrew libomp at: $LIBOMP_PREFIX"
                # Add OpenMP include and lib paths to help CMake find it
                export CFLAGS="-I$LIBOMP_PREFIX/include $CFLAGS"
                export CXXFLAGS="-I$LIBOMP_PREFIX/include $CXXFLAGS"
                export LDFLAGS="-L$LIBOMP_PREFIX/lib $LDFLAGS"
                # Tell CMake where to find OpenMP
                CMAKE_OPTIONS="$CMAKE_OPTIONS -DUSE_OPENMP=ON -DOpenMP_C_FLAGS=-I$LIBOMP_PREFIX/include -DOpenMP_CXX_FLAGS=-I$LIBOMP_PREFIX/include -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=$LIBOMP_PREFIX/lib/libomp.dylib"
            else
                echo "Homebrew libomp not found, trying standard locations"
                # Try standard Homebrew locations
                for loc in "/usr/local/opt/libomp" "/opt/homebrew/opt/libomp"; do
                    if [[ -d "$loc" ]]; then
                        echo "Found libomp at: $loc"
                        export CFLAGS="-I$loc/include $CFLAGS"
                        export CXXFLAGS="-I$loc/include $CXXFLAGS"
                        export LDFLAGS="-L$loc/lib $LDFLAGS"
                        CMAKE_OPTIONS="$CMAKE_OPTIONS -DUSE_OPENMP=ON -DOpenMP_C_FLAGS=-I$loc/include -DOpenMP_CXX_FLAGS=-I$loc/include -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=$loc/lib/libomp.dylib"
                        break
                    fi
                done
            fi
        else
            echo "Homebrew not found, using CMake's OpenMP detection"
            CMAKE_OPTIONS="$CMAKE_OPTIONS -DUSE_OPENMP=ON"
        fi
    else
        echo "Disabling OpenMP support on macOS"
        CMAKE_OPTIONS="$CMAKE_OPTIONS -DUSE_OPENMP=OFF"
    fi
else
    # Linux and other platforms - OpenMP is generally well-supported
    echo "Building for Linux or other platform"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DUSE_OPENMP=ON"
fi

# Configure XGBoost build
echo "CMake options: $CMAKE_OPTIONS"
cmake $CMAKE_OPTIONS .

# Build the objxgboost target which compiles all the object files
make objxgboost

# Make sure library directory exists
mkdir -p ../lib

# Create the static library directly using ar
find src/CMakeFiles/objxgboost.dir -name "*.o" | xargs ar rcs ../lib/libxgboost.a

# Copy dmlc library
cp dmlc-core/libdmlc.a ../lib/

# Check if the library was created
if [ -f "../lib/libxgboost.a" ]; then
    echo "Successfully built libxgboost.a"
    exit 0
else
    echo "Failed to build libxgboost.a"
    exit 1
fi 