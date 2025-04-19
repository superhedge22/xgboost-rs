extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let xgb_root = env::current_dir().unwrap().join("xgboost");

    // Print the current version of XGBoost that we're using
    println!("cargo:warning=Building with XGBoost 3.0.0");

    // Set USE_OPENMP to 1 by default on macOS to ensure it's built with OpenMP support
    if env::var("USE_OPENMP").is_err() && target.contains("apple") {
        println!("cargo:warning=Setting USE_OPENMP=1 by default on macOS");
        env::set_var("USE_OPENMP", "1");
    }

    // Ensure XGBoost is built
    if !Path::new(&xgb_root.join("lib/libxgboost.a")).exists() {
        println!("cargo:warning=Building XGBoost from source");
        
        // Pass environment variables to the build script if needed
        let build_script = env::current_dir().unwrap().join("build_xgboost.sh");
        
        // Allow passing USE_OPENMP through environment
        let mut cmd = Command::new(&build_script);
        
        // Pass through the USE_OPENMP environment variable if set
        if let Ok(openmp_val) = env::var("USE_OPENMP") {
            cmd.env("USE_OPENMP", openmp_val);
        }
        
        let status = cmd.status().expect("Failed to execute build script");
            
        if !status.success() {
            panic!("Failed to build XGBoost");
        }
    } else {
        println!("cargo:warning=Using pre-built XGBoost");
    }

    let header_path = xgb_root.join("include/xgboost/c_api.h");
    println!("cargo:warning=Using C API header at: {}", header_path.display());

    // Generate bindings with very specific settings for the XGBoost C API
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        // Only include XGBoost C API functions and types (no C++ stuff)
        .allowlist_function("XG.*")
        .allowlist_function("DMatrix.*")
        .allowlist_function("Booster.*")
        .allowlist_function("Rabit.*")
        .allowlist_type("XG.*")
        .allowlist_type("bst_.*")
        .allowlist_var("XG.*")
        // Explicitly skip all std:: and other problematic types
        .blocklist_type("std::.*")
        .blocklist_type("__.*")
        // Don't do layout tests which can be problematic with C++ types
        .layout_tests(false)
        // Generate Rustified enums
        .rustified_enum(".*")
        .clang_args(&["-x", "c"])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // Configure the linker - we know exactly where to find the libraries now
    let lib_dir = xgb_root.join("lib");
    println!("cargo:rustc-link-search={}", lib_dir.display());
    
    println!("cargo:rustc-link-lib=static=xgboost");
    println!("cargo:rustc-link-lib=static=dmlc");

    // Handle platform-specific linking
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
        
        // On macOS, check if OpenMP was enabled - default is now TRUE
        let use_openmp = env::var("USE_OPENMP").unwrap_or_else(|_| "1".to_string());
        
        if use_openmp == "1" {
            println!("cargo:warning=Linking with OpenMP on macOS");
            
            // Link against libomp when OpenMP is enabled
            println!("cargo:rustc-link-lib=dylib=omp");
            
            // First check for homebrew-installed libomp
            if let Ok(brew_prefix) = Command::new("brew")
                .args(&["--prefix", "libomp"])
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            {
                if !brew_prefix.is_empty() {
                    println!("cargo:warning=Found Homebrew libomp at: {}", brew_prefix);
                    println!("cargo:rustc-link-search={}/lib", brew_prefix);
                    
                    // Also set LIBRARY_PATH for runtime linking
                    println!("cargo:rustc-env=LIBRARY_PATH={}/lib:$LIBRARY_PATH", brew_prefix);
                    println!("cargo:rustc-env=DYLD_LIBRARY_PATH={}/lib:$DYLD_LIBRARY_PATH", brew_prefix);
                }
            }
            
            // Also try the default Homebrew location as fallback
            println!("cargo:rustc-link-search=/usr/local/opt/libomp/lib");
            println!("cargo:rustc-link-search=/opt/homebrew/opt/libomp/lib");
        } else {
            println!("cargo:warning=OpenMP support is disabled");
        }
    } else if target.contains("linux") {
        // Linux - use appropriate C++ standard library
        println!("cargo:rustc-link-lib=dylib=stdc++");
        
        // For ARM vs x86_64, both should work with gomp
        println!("cargo:rustc-link-lib=dylib=gomp");
    } else if target.contains("windows") {
        // Windows linking
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // Windows OpenMP is usually part of MSVC so no separate link needed
    } else {
        // Default case for other targets
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
    
    // Force rustc to re-run this build script if the xgboost checkout changes
    println!("cargo:rerun-if-changed=xgboost");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build_xgboost.sh");
    println!("cargo:rerun-if-env-changed=USE_OPENMP");
}
