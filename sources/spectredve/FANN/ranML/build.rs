//! Build script for ranML
//! 
//! This script generates build information and version data
//! that is embedded into the compiled binary.

use std::env;

fn main() {
    // Set default build information
    println!("cargo:rustc-env=VERGEN_GIT_SHA=unknown");
    println!("cargo:rustc-env=VERGEN_BUILD_DATE=unknown");

    // Feature detection
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Detect WASM target
    if let Ok(target) = env::var("TARGET") {
        if target.contains("wasm32") {
            println!("cargo:rustc-cfg=target_wasm");
        }
    }
    
    // Detect features
    if cfg!(feature = "neural") {
        println!("cargo:rustc-cfg=feature_neural");
    }
    
    if cfg!(feature = "forecasting") {
        println!("cargo:rustc-cfg=feature_forecasting");
    }
    
    if cfg!(feature = "training") {
        println!("cargo:rustc-cfg=feature_training");
    }
}