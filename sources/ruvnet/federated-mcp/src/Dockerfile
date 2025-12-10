FROM denoland/deno:1.40.2

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Cache dependencies
RUN deno cache src/apps/deno/server.ts

# Compile the server with all required permissions
RUN deno compile \
    --allow-net \
    --allow-env \
    --allow-read \
    --allow-write \
    --allow-run \
    src/apps/deno/server.ts

# The port that your application listens to
EXPOSE 3000

# Create and use non-root user
RUN chown -R deno:deno /app
USER deno

# Start the server
CMD ["./server"]
