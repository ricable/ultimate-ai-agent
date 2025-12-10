FROM denoland/deno:1.40.2

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY deno.json* ./
COPY import_map.json* ./

# Copy source code
COPY . .

# Install dependencies and cache
RUN deno cache src/apps/deno/server.ts

# Build the application
RUN deno compile \
    --allow-net \
    --allow-env \
    --allow-read \
    --allow-write \
    --allow-run \
    src/apps/deno/server.ts

# Create production image
FROM denoland/deno:1.40.2-slim

WORKDIR /app

# Copy the compiled binary
COPY --from=0 /app/server ./

# The port that your application listens to
EXPOSE 3000

# Create and use non-root user
RUN chown -R deno:deno /app
USER deno

# Start the server
CMD ["./server"]
