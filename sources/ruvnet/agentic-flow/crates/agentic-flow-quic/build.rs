use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();

    // WASM-specific configuration
    if target.contains("wasm32") {
        println!("cargo:rustc-cfg=wasm");

        // Optimize for size in WASM builds
        println!("cargo:rustc-env=RUSTFLAGS=-C opt-level=z");

        // Enable WASM features
        println!("cargo:rustc-cfg=feature=\"wasm\"");
    }

    // Enable platform-specific optimizations
    if target.contains("x86_64") {
        println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
