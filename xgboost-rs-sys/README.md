# xgboost-rs-sys

Low-level Rust bindings for XGBoost 3.0.0.

## Important Note for XGBoost 3.0.0 on macOS

With the upgrade to XGBoost 3.0.0, **OpenMP is now required** for proper operation on macOS. The library will be built with OpenMP support by default. Please ensure you have `libomp` installed:

```bash
# Install the dependency (requires Homebrew)
./install_deps.sh

# Or manually
brew install libomp
```

When running tests or applications, you may need to set these environment variables:

```bash
export LIBRARY_PATH="$(brew --prefix libomp)/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

## Building on Different Platforms

### macOS (Apple Silicon & Intel)

On macOS with XGBoost 3.0.0, OpenMP is now enabled by default as it's required for proper operation. 

1. Install OpenMP via Homebrew (required):
   ```
   brew install libomp
   ```

2. To explicitly control OpenMP support:
   ```
   # Enable OpenMP (default)
   USE_OPENMP=1 cargo build
   
   # Disable OpenMP (not recommended for XGBoost 3.0.0)
   USE_OPENMP=0 cargo build
   ```

### Linux (ARM & x86_64)

On Linux, OpenMP is enabled by default as it is well-supported. No special configuration is needed.

### Platform Specific Notes

#### Apple Silicon (M1/M2/M3)

With XGBoost 3.0.0, OpenMP is now required even on Apple Silicon. Make sure you have libomp installed.

#### macOS Intel

OpenMP is required for XGBoost 3.0.0. Install libomp as described above.

#### Linux ARM64

The build has been tested on various ARM64 platforms and should work without issues.

#### Linux x86_64

This is the most common platform and is well-supported.

## Environment Variables

- `USE_OPENMP`: Set to `1` to enable OpenMP support (default), `0` to disable it (not recommended for XGBoost 3.0.0).
- `LIBRARY_PATH`: May need to include the path to libomp library on macOS.
- `DYLD_LIBRARY_PATH`: May need to include the path to libomp library on macOS.

## Troubleshooting

### Running Tests on macOS

If you encounter issues running tests on macOS with errors about missing OpenMP symbols like `___kmpc_barrier`:

1. Ensure libomp is installed:
   ```
   brew install libomp
   ```

2. Set the necessary environment variables:
   ```
   export LIBRARY_PATH="$(brew --prefix libomp)/lib:$LIBRARY_PATH"
   export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
   ```

3. Run the tests again:
   ```
   cargo test
   ```

### Linking Errors

If you encounter linking errors:

1. Check that you have a C++ compiler installed
2. On macOS, ensure libomp is installed via Homebrew
3. On Linux, ensure the appropriate development packages are installed:
   ```
   # On Debian/Ubuntu
   apt-get install build-essential libgomp1
   
   # On Fedora/RHEL
   dnf install gcc-c++ libgomp
   ```

## Dependencies
- See [XGBoost Build from Source](https://xgboost.readthedocs.io/en/stable/build.html#id4)
- CMake 3.14 or higher
- A recent C++ compiler supporting C++11 (g++-5.0 or higher)
- libclang
- OpenMP (required for XGBoost 3.0.0)

## Preprocessing Modules

This library provides scikit-learn inspired preprocessing modules to handle data preparation before feeding it to XGBoost models:

### Available Transformers

- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance
- **SimpleImputer**: Handles missing values using strategies like mean, median, most frequent, or constant values
- **OneHotEncoder**: Transforms categorical features into one-hot encoded numerical features
- **ColumnTransformer**: Applies different transformers to specific columns of a dataset
- **Pipeline**: Chains multiple transformers into a single transformation workflow with optional prediction

### Example Usage

```rust
use ndarray::{array, ArrayView2};
use xgboostrs::preprocessing::{
    pipeline::Pipeline,
    imputer::SimpleImputer,
    scaler::StandardScaler,
    Transformer
};
use xgboostrs::parameters::preprocessing::ImputationStrategy;

// Create preprocessing steps
let imputer = SimpleImputer::new(ImputationStrategy::Mean);
let scaler = StandardScaler::new();

// Create a pipeline with multiple transformers
let steps = vec![
    ("imputer".to_string(), Box::new(imputer) as Box<dyn Transformer>),
    ("scaler".to_string(), Box::new(scaler) as Box<dyn Transformer>),
];

let mut pipeline = Pipeline::new(steps);

// Sample data with missing values
let data = array![[1.0, f64::NAN], [3.0, 2.0], [f64::NAN, 5.0]];

// Fit and transform the data
let transformed = pipeline.fit_transform(&data.view()).unwrap();

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
