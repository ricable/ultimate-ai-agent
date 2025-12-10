use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Set target-specific optimizations
    if let Ok(target) = env::var("TARGET") {
        if target.contains("x86_64") {
            // Enable AVX2 for x86_64 targets
            println!("cargo:rustc-cfg=target_feature=\"avx2\"");
            println!("cargo:rustc-cfg=target_feature=\"fma\"");
        }
        
        if target.contains("aarch64") {
            // Enable NEON for ARM64 targets
            println!("cargo:rustc-cfg=target_feature=\"neon\"");
        }
    }
    
    // Link against system BLAS/LAPACK libraries
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=blas");
        println!("cargo:rustc-link-lib=lapack");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    
    // Enable link-time optimization for release builds
    if env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-link-arg=-flto");
    }
}