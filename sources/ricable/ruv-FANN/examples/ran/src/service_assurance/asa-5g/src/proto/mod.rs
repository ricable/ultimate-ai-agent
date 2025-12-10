pub mod ran {
    pub mod service_assurance {
        tonic::include_proto!("ran.service_assurance");
    }
}

pub use ran::service_assurance::*;