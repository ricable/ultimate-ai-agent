use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const JJ_VERSION: &str = "0.35.0";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Skip jj binary download for docs.rs builds
    if env::var("DOCS_RS").is_ok() {
        println!("cargo:warning=Skipping jj binary download for docs.rs build");
        create_dummy_binary(&out_dir);
        return;
    }

    // Determine the jj binary platform name
    // Note: jj only provides musl builds for Linux, use them for both gnu and musl targets
    let platform_info = match target.as_str() {
        "x86_64-apple-darwin" => Some(("jj-v{}-x86_64-apple-darwin.tar.gz", "jj")),
        "aarch64-apple-darwin" => Some(("jj-v{}-aarch64-apple-darwin.tar.gz", "jj")),
        "x86_64-pc-windows-msvc" => Some(("jj-v{}-x86_64-pc-windows-msvc.zip", "jj.exe")),
        "aarch64-pc-windows-msvc" => Some(("jj-v{}-aarch64-pc-windows-msvc.zip", "jj.exe")),
        "x86_64-unknown-linux-gnu" | "x86_64-unknown-linux-musl" =>
            Some(("jj-v{}-x86_64-unknown-linux-musl.tar.gz", "jj")),
        "aarch64-unknown-linux-gnu" | "aarch64-unknown-linux-musl" =>
            Some(("jj-v{}-aarch64-unknown-linux-musl.tar.gz", "jj")),
        _ => {
            println!("cargo:warning=Unsupported platform for jj binary bundling: {}", target);
            create_dummy_binary(&out_dir);
            return;
        }
    };

    if let Some((archive_name, binary_path)) = platform_info {
        let archive_name = archive_name.replace("{}", JJ_VERSION);
        let binary_path = binary_path.replace("{}", JJ_VERSION);

        match download_and_extract_jj(&out_dir, &archive_name, &binary_path) {
            Ok(jj_binary) => {
                println!("cargo:warning=Successfully downloaded jj binary for {}", target);

                // Copy to standard location for embedding
                let dest = out_dir.join("jj");
                fs::copy(&jj_binary, &dest).expect("Failed to copy jj binary");

                // Make executable on Unix
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = fs::metadata(&dest).unwrap().permissions();
                    perms.set_mode(0o755);
                    fs::set_permissions(&dest, perms).unwrap();
                }
            }
            Err(e) => {
                println!("cargo:warning=Failed to download jj binary: {}. Using fallback to system jj.", e);
                create_dummy_binary(&out_dir);
            }
        }
    }
}

fn download_and_extract_jj(out_dir: &Path, archive_name: &str, binary_path: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = out_dir.join("jj-cache");
    fs::create_dir_all(&cache_dir)?;

    let archive_path = cache_dir.join(archive_name);

    // Download if not cached
    if !archive_path.exists() {
        let url = format!(
            "https://github.com/jj-vcs/jj/releases/download/v{}/{}",
            JJ_VERSION, archive_name
        );

        println!("cargo:warning=Downloading jj from {}", url);

        let response = reqwest::blocking::get(&url)?;
        if !response.status().is_success() {
            return Err(format!("Failed to download jj: HTTP {}", response.status()).into());
        }

        let mut file = fs::File::create(&archive_path)?;
        let content = response.bytes()?;
        file.write_all(&content)?;

        println!("cargo:warning=Downloaded {} bytes", content.len());
    } else {
        println!("cargo:warning=Using cached jj archive");
    }

    // Extract archive (tar.gz or zip)
    if archive_path.extension().and_then(|s| s.to_str()) == Some("zip") {
        // Extract ZIP (Windows builds)
        let file = fs::File::open(&archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        // Extract the jj binary
        let mut jj_file = archive.by_name(binary_path)?;
        let jj_binary = cache_dir.join(binary_path);

        let mut out_file = fs::File::create(&jj_binary)?;
        std::io::copy(&mut jj_file, &mut out_file)?;

        Ok(jj_binary)
    } else {
        // Extract tar.gz (Unix builds)
        let tar_gz = fs::File::open(&archive_path)?;
        let tar = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(tar);

        // Extract to cache directory
        archive.unpack(&cache_dir)?;

        // Find the jj binary
        let jj_binary = cache_dir.join(binary_path);
        if !jj_binary.exists() {
            return Err(format!("jj binary not found at expected path: {:?}", jj_binary).into());
        }

        Ok(jj_binary)
    }
}

fn create_dummy_binary(out_dir: &Path) {
    // Create a placeholder file indicating jj binary should be found in PATH
    let placeholder = out_dir.join("jj");
    fs::write(placeholder, b"SYSTEM_JJ").expect("Failed to create placeholder");
}
