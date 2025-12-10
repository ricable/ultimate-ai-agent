use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::num::NonZeroU32;
use std::sync::LazyLock;

// Global rate limiter: 100 requests per minute
static RATE_LIMITER: LazyLock<RateLimiter<NotKeyed, InMemoryState, DefaultClock>> =
    LazyLock::new(|| {
        let quota = Quota::per_minute(NonZeroU32::new(100).unwrap());
        RateLimiter::direct(quota)
    });

pub async fn rate_limit_middleware(
    request: Request,
    next: Next,
) -> Result<Response, RateLimitError> {
    // Check rate limit
    match RATE_LIMITER.check() {
        Ok(_) => {
            metrics::counter!("api.rate_limit.allowed").increment(1);
            Ok(next.run(request).await)
        }
        Err(_) => {
            metrics::counter!("api.rate_limit.exceeded").increment(1);
            Err(RateLimitError)
        }
    }
}

#[derive(Debug)]
pub struct RateLimitError;

impl IntoResponse for RateLimitError {
    fn into_response(self) -> Response {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded. Please try again later.",
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_exists() {
        // Access the rate limiter to ensure it initializes
        let _limiter = &*RATE_LIMITER;
    }
}
