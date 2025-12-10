use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use metrics_exporter_prometheus::PrometheusHandle;

/// GET /api/metrics
pub async fn metrics_handler(
    State(handle): State<PrometheusHandle>,
) -> Response {
    let metrics = handle.render();

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        metrics,
    )
        .into_response()
}
