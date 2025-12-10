//! Optimized build script for CUDA-Rust-WASM project
//! Supports cross-platform compilation with caching and performance optimizations

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let start_time = std::time::Instant::now();
    
    // Get build environment info
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=bindings/");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=OPENCL_ROOT");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    
    // Enable parallel compilation if supported
    if env::var("CARGO_FEATURE_PARALLEL_COMPILATION").is_ok() {
        if let Ok(num_jobs) = env::var("NUM_JOBS") {
            println!("cargo:rustc-env=RAYON_NUM_THREADS={num_jobs}");
        }
    }
    
    // Configure for WASM target with enhanced optimizations
    if target_arch == "wasm32" {
        configure_wasm_build(&target_env, &profile);
    } else {
        configure_native_build(&target_os, &target_arch, &profile);
    }
    
    // Configure GPU backends
    configure_gpu_backends(&target_os, &target_arch);
    
    // Generate bindings if needed
    #[cfg(feature = "native-bindings")]
    {
        generate_native_bindings(&out_dir);
    }
    
    // Performance and caching optimizations
    configure_build_optimizations(&profile, &target_arch);
    
    // Build time reporting
    let build_time = start_time.elapsed();
    if build_time.as_millis() > 1000 {
        println!("cargo:warning=Build configuration took {:.2}s", build_time.as_secs_f64());
    }
}

fn configure_wasm_build(target_env: &str, profile: &str) {
    println!("cargo:rustc-cfg=wasm_target");
    
    // Enhanced WASM optimizations
    println!("cargo:rustc-env=WASM_BINDGEN_WEAKREF=1");
    println!("cargo:rustc-env=WASM_BINDGEN_EXTERNREF_XFORM=1");
    
    // Enable SIMD if supported
    if env::var("CARGO_FEATURE_WASM_SIMD").is_ok() {
        println!("cargo:rustc-cfg=wasm_simd");
        println!("cargo:rustc-target-feature=+simd128");
    }
    
    // WASM-specific link arguments for size optimization
    if profile == "release" || profile == "wasm" {
        println!("cargo:rustc-link-arg=--no-entry");
        println!("cargo:rustc-link-arg=--gc-sections");
        println!("cargo:rustc-link-arg=--strip-all");
        println!("cargo:rustc-link-arg=-z");
        println!("cargo:rustc-link-arg=stack-size=1048576"); // 1MB stack
        
        // Enable bulk memory operations
        println!("cargo:rustc-link-arg=--enable-bulk-memory");
        println!("cargo:rustc-link-arg=--enable-mutable-globals");
        
        // Size optimizations
        if profile == "wasm" {
            println!("cargo:rustc-link-arg=-O3");
            println!("cargo:rustc-link-arg=--lto-O3");
        }
    }
    
    // Web-specific features
    if target_env == "unknown" {
        println!("cargo:rustc-cfg=web_target");
    }
}

fn configure_native_build(target_os: &str, target_arch: &str, profile: &str) {
    println!("cargo:rustc-cfg=native_target");
    
    // Platform-specific optimizations
    match target_os {
        "windows" => {
            println!("cargo:rustc-link-lib=dylib=kernel32");
            println!("cargo:rustc-link-lib=dylib=user32");
            println!("cargo:rustc-link-lib=dylib=shell32");
            if profile == "release" {
                println!("cargo:rustc-link-arg=/LTCG"); // Link-time code generation
            }
        },
        "macos" => {
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            if target_arch == "aarch64" {
                println!("cargo:rustc-cfg=apple_silicon");
            }
        },
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
            if profile == "release" {
                println!("cargo:rustc-link-arg=-Wl,--gc-sections");
                println!("cargo:rustc-link-arg=-Wl,--strip-all");
            }
        },
        _ => {}
    }
    
    // Architecture-specific optimizations
    match target_arch {
        "x86_64" => {
            println!("cargo:rustc-cfg=x86_64_target");
            if env::var("CARGO_FEATURE_OPTIMIZED_BUILD").is_ok() {
                println!("cargo:rustc-target-feature=+avx2,+fma");
            }
        },
        "aarch64" => {
            println!("cargo:rustc-cfg=aarch64_target");
            if env::var("CARGO_FEATURE_OPTIMIZED_BUILD").is_ok() {
                println!("cargo:rustc-target-feature=+neon");
            }
        },
        _ => {}
    }
}

fn configure_gpu_backends(target_os: &str, target_arch: &str) {
    // CUDA backend configuration
    #[cfg(feature = "cuda-backend")]
    {
        if let Some(cuda_path) = find_cuda_installation() {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path.display());
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path.display());
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=curand");
            println!("cargo:rustc-cfg=has_cuda");
            
            // CUDA version detection
            if let Some(version) = detect_cuda_version(&cuda_path) {
                println!("cargo:rustc-env=CUDA_VERSION={}", version);
                if version >= 11.0 {
                    println!("cargo:rustc-cfg=cuda_11_plus");
                }
            }
        } else {
            println!("cargo:warning=CUDA not found, CUDA backend disabled");
        }
    }
    
    // OpenCL backend configuration
    #[cfg(feature = "opencl-backend")]
    {
        if find_opencl_installation().is_some() {
            println!("cargo:rustc-cfg=has_opencl");
            match target_os {
                "windows" => println!("cargo:rustc-link-lib=OpenCL"),
                "macos" => println!("cargo:rustc-link-lib=framework=OpenCL"),
                "linux" => println!("cargo:rustc-link-lib=OpenCL"),
                _ => {}
            }
        }
    }
    
    // Vulkan backend configuration
    #[cfg(feature = "vulkan")]
    {
        if let Some(vulkan_path) = find_vulkan_installation() {
            println!("cargo:rustc-link-search=native={}/lib", vulkan_path.display());
            println!("cargo:rustc-link-lib=vulkan");
            println!("cargo:rustc-cfg=has_vulkan");
        }
    }
}

fn configure_build_optimizations(profile: &str, target_arch: &str) {
    // Link-time optimizations
    if profile == "release" {
        println!("cargo:rustc-env=RUST_LTO=fat");
        
        // Enable additional optimizations for release builds
        if target_arch == "wasm32" {
            println!("cargo:rustc-env=WASM_OPT_LEVEL=3");
        } else {
            // Native optimizations
            println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
        }
    }
    
    // Build caching optimizations
    if let Ok(cache_dir) = env::var("CARGO_TARGET_DIR") {
        println!("cargo:rustc-env=CARGO_BUILD_CACHE={cache_dir}");
    }
    
    // Incremental compilation for development
    if profile == "dev" {
        println!("cargo:rustc-env=CARGO_INCREMENTAL=1");
    }
}

// Helper functions for GPU backend detection

fn find_cuda_installation() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(cuda_path);
        if path.exists() {
            return Some(path);
        }
    }
    
    // Common CUDA installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7",
    ];
    
    for path_str in &common_paths {
        let path = PathBuf::from(path_str);
        if path.exists() && path.join("lib64").exists() {
            return Some(path);
        }
    }
    
    // Try pkg-config
    if Command::new("pkg-config").args(["--exists", "cuda"]).status().map(|s| s.success()).unwrap_or(false) {
        if let Ok(output) = Command::new("pkg-config").args(["--variable=cudaroot", "cuda"]).output() {
            let path_str = String::from_utf8_lossy(&output.stdout);
            let path_str = path_str.trim();
            let path = PathBuf::from(path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }
    
    None
}

fn detect_cuda_version(cuda_path: &Path) -> Option<f32> {
    let nvcc_path = cuda_path.join("bin").join("nvcc");
    if let Ok(output) = Command::new(nvcc_path).args(["--version"]).output() {
        let version_str = String::from_utf8_lossy(&output.stdout);
        // Parse version from output like "Cuda compilation tools, release 11.8, V11.8.89"
        // Simple string parsing without regex
        if let Some(release_pos) = version_str.find("release ") {
            let version_part = &version_str[release_pos + 8..];
            if let Some(comma_pos) = version_part.find(',') {
                let version_num = &version_part[..comma_pos];
                if let Ok(version) = version_num.parse::<f32>() {
                    return Some(version);
                }
            }
        }
    }
    None
}

fn find_opencl_installation() -> Option<PathBuf> {
    // Check environment variable
    if let Ok(opencl_root) = env::var("OPENCL_ROOT") {
        let path = PathBuf::from(opencl_root);
        if path.exists() {
            return Some(path);
        }
    }
    
    // Platform-specific paths
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "windows" => {
            // Windows: Check for Intel OpenCL SDK or AMD APP SDK
            let paths = [
                "C:\\Program Files\\Intel\\OpenCL SDK",
                "C:\\Program Files (x86)\\Intel\\OpenCL SDK",
                "C:\\Program Files\\AMD APP SDK",
            ];
            for path_str in &paths {
                let path = PathBuf::from(path_str);
                if path.exists() {
                    return Some(path);
                }
            }
        },
        "macos" => {
            // macOS has OpenCL framework built-in
            let framework_path = PathBuf::from("/System/Library/Frameworks/OpenCL.framework");
            if framework_path.exists() {
                return Some(framework_path);
            }
        },
        "linux" => {
            // Linux: Check common installation paths
            let paths = [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib64",
                "/usr/local/lib",
                "/opt/intel/opencl",
            ];
            for path_str in &paths {
                let path = PathBuf::from(path_str);
                if path.join("libOpenCL.so").exists() {
                    return Some(path);
                }
            }
        },
        _ => {}
    }
    
    None
}

fn find_vulkan_installation() -> Option<PathBuf> {
    // Check environment variable
    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
        let path = PathBuf::from(vulkan_sdk);
        if path.exists() {
            return Some(path);
        }
    }
    
    // Try pkg-config
    if Command::new("pkg-config").args(["--exists", "vulkan"]).status().map(|s| s.success()).unwrap_or(false) {
        if let Ok(output) = Command::new("pkg-config").args(["--variable=libdir", "vulkan"]).output() {
            let path_str = String::from_utf8_lossy(&output.stdout);
            let path_str = path_str.trim();
            let path = PathBuf::from(path_str);
            if path.exists() {
                return Some(path.parent()?.to_path_buf());
            }
        }
    }
    
    None
}

#[cfg(feature = "native-bindings")]
fn generate_native_bindings(out_dir: &Path) {
    let header_path = "src/backend/native/cuda_wrapper.h";
    
    // Only generate if header exists
    if !Path::new(header_path).exists() {
        println!("cargo:warning=Header file {} not found, skipping binding generation", header_path);
        return;
    }
    
    let bindings = bindgen::Builder::default()
        .header(header_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    
    bindings
        .write_to_file(out_dir.join("cuda_bindings.rs"))
        .expect("Couldn't write bindings!");
    
    println!("cargo:rerun-if-changed={}", header_path);
}