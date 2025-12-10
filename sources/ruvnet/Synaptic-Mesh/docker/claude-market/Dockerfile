# Multi-stage build for minimal Claude container
FROM alpine:latest as builder

# Install minimal dependencies
RUN apk add --no-cache curl ca-certificates

# Create non-root user
RUN adduser -D -s /bin/sh claude

FROM scratch

# Copy CA certificates and user info
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /tmp /tmp

# In production, copy the actual claude binary here
# COPY --from=builder /usr/local/bin/claude /usr/local/bin/claude

# Create minimal entrypoint
COPY entrypoint.sh /entrypoint.sh

USER claude
WORKDIR /tmp

ENTRYPOINT ["/entrypoint.sh"]