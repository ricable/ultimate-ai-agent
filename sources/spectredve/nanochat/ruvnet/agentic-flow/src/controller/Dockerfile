# Build stage
FROM golang:1.21 as builder

WORKDIR /workspace

# Copy go mod files
COPY go.mod go.mod
COPY go.sum go.sum

# Cache deps before building and copying source
RUN go mod download

# Copy the go source
COPY cmd/ cmd/
COPY api/ api/
COPY internal/ internal/
COPY pkg/ pkg/

# Build
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o manager cmd/manager/main.go

# Final stage
FROM alpine:3.18

# Install jj (Jujutsu)
RUN apk add --no-cache ca-certificates git && \
    wget https://github.com/martinvonz/jj/releases/download/v0.12.0/jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz && \
    mv jj /usr/local/bin/ && \
    rm jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz && \
    chmod +x /usr/local/bin/jj

WORKDIR /

# Copy the binary from builder
COPY --from=builder /workspace/manager .

# Use non-root user
USER 65532:65532

ENTRYPOINT ["/manager"]
