{
  "targets": [
    {
      "target_name": "cuda_rust_wasm",
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "cflags": [ "-O3", "-ffast-math", "-march=native" ],
      "cflags_cc": [ "-O3", "-ffast-math", "-march=native", "-std=c++17" ],
      "sources": [ 
        "src/cuda_rust_wasm.cc",
        "src/transpiler.cc",
        "src/runtime.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../target/release",
        "../../src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "libraries": [
        "-L../../target/release",
        "-lcuda_rust_wasm"
      ],
      "defines": [
        "NAPI_VERSION=8",
        "NODE_ADDON_API_DISABLE_DEPRECATED",
        "CUDA_WASM_OPTIMIZED"
      ],
      "conditions": [
        ["OS=='win'", {
          "libraries": [
            "-lws2_32",
            "-luserenv",
            "-ladvapi32",
            "-lkernel32"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "Optimization": 3,
              "FavorSizeOrSpeed": 1,
              "InlineFunctionExpansion": 2,
              "WholeProgramOptimization": "true",
              "OmitFramePointers": "true",
              "EnableFunctionLevelLinking": "true",
              "RuntimeLibrary": 2
            },
            "VCLinkerTool": {
              "LinkTimeCodeGeneration": 1,
              "OptimizeReferences": 2,
              "EnableCOMDATFolding": 2
            }
          }
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "10.15",
            "GCC_OPTIMIZATION_LEVEL": "3",
            "LLVM_LTO": "YES",
            "GCC_GENERATE_DEBUGGING_SYMBOLS": "NO",
            "DEPLOYMENT_POSTPROCESSING": "YES",
            "STRIP_INSTALLED_PRODUCT": "YES",
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "OTHER_CPLUSPLUSFLAGS": [
              "-ffast-math",
              "-march=native",
              "-mtune=native"
            ]
          },
          "libraries": [
            "-framework Accelerate",
            "-framework CoreFoundation"
          ]
        }],
        ["OS=='linux'", {
          "cflags": [ "-flto", "-fuse-linker-plugin" ],
          "cflags_cc": [ "-flto", "-fuse-linker-plugin" ],
          "ldflags": [ "-flto", "-Wl,--gc-sections", "-Wl,--strip-all" ],
          "libraries": [
            "-lpthread",
            "-ldl",
            "-lm"
          ]
        }],
        ["target_arch=='x64'", {
          "cflags": [ "-msse4.2", "-mavx", "-mavx2" ],
          "cflags_cc": [ "-msse4.2", "-mavx", "-mavx2" ],
          "defines": [ "CUDA_WASM_X64_OPTIMIZED" ]
        }],
        ["target_arch=='arm64'", {
          "cflags": [ "-mcpu=native" ],
          "cflags_cc": [ "-mcpu=native" ],
          "defines": [ "CUDA_WASM_ARM64_OPTIMIZED" ]
        }]
      ],
      "configurations": {
        "Release": {
          "cflags": [ "-O3", "-DNDEBUG" ],
          "cflags_cc": [ "-O3", "-DNDEBUG" ]
        },
        "Debug": {
          "cflags": [ "-g", "-O0" ],
          "cflags_cc": [ "-g", "-O0" ],
          "defines": [ "DEBUG", "CUDA_WASM_DEBUG" ]
        }
      }
    }
  ]
}