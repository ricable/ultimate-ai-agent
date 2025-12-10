# Publishing cuda-wasm Crate

Since you cannot rename a crate on crates.io, here's how to publish a new `cuda-wasm` crate:

## Option 1: Publish cuda-wasm as a New Crate

1. Create a new directory outside the workspace:
```bash
cd /tmp
mkdir cuda-wasm-publish
cd cuda-wasm-publish
```

2. Copy the necessary files:
```bash
cp -r /workspaces/ruv-FANN/cuda-wasm/src .
cp /workspaces/ruv-FANN/cuda-wasm/Cargo-cuda-wasm.toml ./Cargo.toml
cp /workspaces/ruv-FANN/cuda-wasm/README.md .
cp /workspaces/ruv-FANN/cuda-wasm/LICENSE .
cp /workspaces/ruv-FANN/cuda-wasm/build.rs .
```

3. Update src/lib.rs to use the content from lib-cuda-wasm.rs:
```bash
cp /workspaces/ruv-FANN/cuda-wasm/src/lib-cuda-wasm.rs ./src/lib.rs
```

4. Add [workspace] to Cargo.toml to avoid workspace conflicts

5. Publish:
```bash
cargo publish
```

## Option 2: Update cuda-rust-wasm Documentation

Add a notice to the cuda-rust-wasm README:

```markdown
> **Note**: For npm/JavaScript users, please use the npm package:
> ```bash
> npm install -g cuda-wasm
> # or
> npx cuda-wasm transpile kernel.cu
> ```
```

## NPM Package Information

The npm package `cuda-wasm` is already published and working:
- Version: 1.1.0
- Install: `npm install -g cuda-wasm`
- Usage: `npx cuda-wasm transpile kernel.cu -o kernel.wasm`

## Recommendation

Since the npm package is already named `cuda-wasm` and working well, I recommend:

1. Keep `cuda-rust-wasm` as the Rust crate name
2. Update the README to clarify:
   - Rust developers: use `cuda-rust-wasm` crate
   - JavaScript/CLI users: use `cuda-wasm` npm package
3. Add cross-references between the two packages

This avoids confusion and leverages the fact that npm and crates.io are separate ecosystems.