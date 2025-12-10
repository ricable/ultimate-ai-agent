use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    
    // Compile the proto files
    tonic_build::configure()
        .file_descriptor_set_path(out_dir.join("service_assurance_descriptor.bin"))
        .compile(
            &["../../shared/proto/proto/service_assurance.proto"],
            &["../../shared/proto/proto"],
        )?;
    
    Ok(())
}