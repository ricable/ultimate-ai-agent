#!/usr/bin/env nu

# Container management utilities for the polyglot development environment
# Usage: nu scripts/containers.nu <command> [options]

use ../common.nu *

# Main container command dispatcher
def main [command?: string] {
    if ($command | is-empty) {
        show-help
    } else {
        match $command {
            "build" => { main build }
            "run" => { main run }
            "status" => { main status }
            "registry" => { main registry }
            "compose" => { main compose }
            "cleanup" => { main cleanup }
            _ => {
                log error $"Unknown command: ($command)"
                show-help
                exit 1
            }
        }
    }
}

def show-help [] {
    log info "Container utilities for polyglot development environment"
    print ""
    print "Commands:"
    print "  build      - Build container images for environments"
    print "  run        - Run containers for development"
    print "  status     - Show container and image status"
    print "  registry   - Manage container registry operations"
    print "  compose    - Docker Compose operations"
    print "  cleanup    - Clean up containers and images"
    print ""
    print "Examples:"
    print "  nu scripts/containers.nu build --env python"
    print "  nu scripts/containers.nu run --env all --detach"
    print "  nu scripts/containers.nu status"
}

# Build container images for environments
def "main build" [
    --environment: string = "all"
    --push = false
    --registry: string = ""
] {
    log info "Building container images..."
    
    let environments = get-container-environments $environment
    
    for environment in $environments {
        build-environment-image $environment $push $registry
    }
}

def get-container-environments [env_filter: string] {
    let all_environments = [
        {name: "python", dir: "python-env", port: 8000, base_image: "python:3.12-slim"},
        {name: "typescript", dir: "typescript-env", port: 3000, base_image: "node:20-alpine"},
        {name: "rust", dir: "rust-env", port: 8080, base_image: "rust:1.70-slim"},
        {name: "go", dir: "go-env", port: 8090, base_image: "golang:1.22-alpine"}
    ]
    
    if $env_filter == "all" {
        $all_environments
    } else {
        $all_environments | where name == $env_filter
    }
}

def build-environment-image [environment: record, push: bool, registry: string] {
    if not ($environment.dir | path exists) {
        log warn $"Environment directory not found: ($environment.dir)"
        return
    }
    
    log info $"Building image for ($environment.name)..."
    
    # Generate Dockerfile if it doesn't exist
    let dockerfile_path = $"($environment.dir)/Dockerfile"
    if not ($dockerfile_path | path exists) {
        generate-dockerfile $environment $dockerfile_path
    }
    
    # Build image
    let image_name = get-image-name $environment.name $registry
    
    try {
        cd $environment.dir
        docker build -t $image_name .
        cd ..
        log success $"✅ Built image: ($image_name)"
        
        if $push and ($registry | is-not-empty) {
            push-image $image_name
        }
        
    } catch { |e|
        log error $"❌ Failed to build ($env.name): ($e.msg)"
    }
}

def generate-dockerfile [environment: record, dockerfile_path: string] {
    log info $"  Generating Dockerfile for ($environment.name)..."
    
    let dockerfile_content = match $environment.name {
        "python" => (generate-python-dockerfile $environment)
        "typescript" => (generate-typescript-dockerfile $environment)
        "rust" => (generate-rust-dockerfile $environment)
        "go" => (generate-go-dockerfile $environment)
        _ => {
            log error $"Unknown environment: ($environment.name)"
            return
        }
    }
    
    $dockerfile_content | save $dockerfile_path --force
    log success $"  ✅ Generated Dockerfile"
}

def generate-python-dockerfile [environment: record] {
    $"FROM ($env.base_image)

WORKDIR /app

# Install devbox
RUN curl -fsSL https://get.jetify.com/devbox | bash

# Copy devbox configuration
COPY devbox.json devbox.lock ./

# Install dependencies
RUN devbox install

# Copy source code
COPY . .

# Install Python dependencies
RUN devbox run install

EXPOSE ($env.port)

CMD [\"devbox\", \"run\", \"start\"]"
}

def generate-typescript-dockerfile [environment: record] {
    $"FROM ($env.base_image)

WORKDIR /app

# Install devbox
RUN apk add --no-cache curl bash \\
    && curl -fsSL https://get.jetify.com/devbox | bash

# Copy devbox configuration
COPY devbox.json devbox.lock ./

# Install dependencies
RUN devbox install

# Copy source code
COPY . .

# Install Node.js dependencies
RUN devbox run install

EXPOSE ($env.port)

CMD [\"devbox\", \"run\", \"start\"]"
}

def generate-rust-dockerfile [environment: record] {
    $"FROM ($env.base_image)

WORKDIR /app

# Install devbox
RUN apt-get update && apt-get install -y curl \\
    && curl -fsSL https://get.jetify.com/devbox | bash

# Copy devbox configuration
COPY devbox.json devbox.lock ./

# Install dependencies
RUN devbox install

# Copy source code
COPY . .

# Build Rust application
RUN devbox run build

EXPOSE ($env.port)

CMD [\"devbox\", \"run\", \"start\"]"
}

def generate-go-dockerfile [environment: record] {
    $"FROM ($env.base_image)

WORKDIR /app

# Install devbox
RUN apk add --no-cache curl bash \\
    && curl -fsSL https://get.jetify.com/devbox | bash

# Copy devbox configuration
COPY devbox.json devbox.lock ./

# Install dependencies
RUN devbox install

# Copy source code
COPY . .

# Build Go application
RUN devbox run build

EXPOSE ($env.port)

CMD [\"devbox\", \"run\", \"start\"]"
}

def get-image-name [env_name: string, registry: string] {
    if ($registry | is-not-empty) {
        $"($registry)/($env_name)-app:latest"
    } else {
        $"($env_name)-app:latest"
    }
}

def push-image [image_name: string] {
    log info $"Pushing image: ($image_name)"
    
    try {
        docker push $image_name
        log success $"✅ Pushed image: ($image_name)"
    } catch { |e|
        log error $"❌ Failed to push image: ($e.msg)"
    }
}

# Run containers for development
def "main run" [
    --environment: string = "all"
    --detach = false
    --port-map = true
] {
    log info "Running containers..."
    
    let environments = get-container-environments $environment
    
    for environment in $environments {
        run-environment-container $environment $detach $port_map
    }
}

def run-environment-container [environment: record, detach: bool, port_map: bool] {
    let image_name = get-image-name $environment.name ""
    let container_name = $"($environment.name)-dev"
    
    # Check if image exists
    let image_exists = try {
        (docker image inspect $image_name | length) > 0
    } catch {
        false
    }
    
    if not $image_exists {
        log warn $"Image not found: ($image_name). Building..."
        build-environment-image $environment false ""
    }
    
    # Stop existing container
    try {
        docker stop $container_name
        docker rm $container_name
    } catch {
        # Container doesn't exist, continue
    }
    
    # Prepare run command
    mut run_args = ["run", "--name", $container_name]
    
    if $detach {
        $run_args = $run_args | append "--detach"
    }
    
    if $port_map {
        $run_args = $run_args | append "--publish" | append $"($environment.port):($environment.port)"
    }
    
    # Add environment variables
    if (".env" | path exists) {
        $run_args = $run_args | append "--env-file" | append ".env"
    }
    
    $run_args = $run_args | append $image_name
    
    log info $"Starting container: ($container_name)"
    
    try {
        docker ...$run_args
        log success $"✅ Started container: ($container_name)"
        
        if $port_map {
            log info $"  Access at: http://localhost:($environment.port)"
        }
    } catch { |e|
        log error $"❌ Failed to start container: ($e.msg)"
    }
}

# Show container and image status
def "main status" [] {
    log info "Container and image status:"
    
    # Show running containers
    log info "Running containers:"
    try {
        let containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | lines | skip 1
        if ($containers | length) > 0 {
            for container in $containers {
                log info $"  ($container)"
            }
        } else {
            log info "  No running containers"
        }
    } catch {
        log warn "  Unable to get container status"
    }
    
    # Show images
    log info "Available images:"
    try {
        let images = docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" 
            | lines 
            | skip 1 
            | where ($it | str contains "-app")
        
        if ($images | length) > 0 {
            for image in $images {
                log info $"  ($image)"
            }
        } else {
            log info "  No polyglot environment images found"
        }
    } catch {
        log warn "  Unable to get image status"
    }
}

# Manage container registry operations
def "main registry" [
    --login = false
    --push-all = false
    --registry: string = ""
] {
    if $login {
        registry-login $registry
    }
    
    if $push_all {
        push-all-images $registry
    }
}

def registry-login [registry: string] {
    let registry_url = if ($registry | is-empty) {
        env get-or-prompt "REGISTRY_SERVER" "Container registry server"
    } else {
        $registry
    }
    
    let username = env get-or-prompt "REGISTRY_USER" "Registry username"
    let password = (env get-or-prompt "REGISTRY_PASSWORD" "Registry password" --secret true)
    
    log info $"Logging into registry: ($registry_url)"
    
    try {
        echo $password | docker login $registry_url --username $username --password-stdin
        log success $"✅ Logged into registry: ($registry_url)"
    } catch { |e|
        log error $"❌ Failed to login to registry: ($e.msg)"
    }
}

def push-all-images [registry: string] {
    log info "Pushing all environment images..."
    
    let environments = get-container-environments "all"
    
    for environment in $environments {
        let local_image = get-image-name $environment.name ""
        let registry_image = get-image-name $environment.name $registry
        
        # Tag for registry
        try {
            docker tag $local_image $registry_image
            push-image $registry_image
        } catch { |e|
            log error $"❌ Failed to push ($environment.name): ($e.msg)"
        }
    }
}

# Docker Compose operations
def "main compose" [
    --generate = false
    --up = false
    --down = false
    --logs = false
] {
    if $generate {
        generate-compose-file
    }
    
    if $up {
        compose-up
    }
    
    if $down {
        compose-down
    }
    
    if $logs {
        compose-logs
    }
}

def generate-compose-file [] {
    log info "Generating docker-compose.yml..."
    
    let environments = get-container-environments "all"
    
    let services = $environments | reduce -f {} { |environment, acc|
        $acc | upsert $environment.name {
            build: $environment.dir
            ports: [$"($environment.port):($environment.port)"]
            environment: ["NODE_ENV=development", "PYTHONPATH=/app/src"]
            volumes: [".env:/app/.env:ro"]
            depends_on: []
        }
    }
    
    let compose_content = {
        version: "3.8"
        services: $services
        networks: {
            polyglot: {
                driver: "bridge"
            }
        }
    }
    
    $compose_content | to yaml | save docker-compose.yml --force
    log success "✅ Generated docker-compose.yml"
}

def compose-up [] {
    log info "Starting all services with Docker Compose..."
    
    try {
        docker-compose up --build --detach
        log success "✅ All services started"
    } catch { |e|
        log error $"❌ Failed to start services: ($e.msg)"
    }
}

def compose-down [] {
    log info "Stopping all services..."
    
    try {
        docker-compose down
        log success "✅ All services stopped"
    } catch { |e|
        log error $"❌ Failed to stop services: ($e.msg)"
    }
}

def compose-logs [] {
    try {
        docker-compose logs --follow
    } catch { |e|
        log error $"❌ Failed to get logs: ($e.msg)"
    }
}

# Clean up containers and images
def "main cleanup" [
    --all = false
    --images = false
    --containers = false
    --volumes = false
] {
    log info "Cleaning up Docker resources..."
    
    if $containers or $all {
        cleanup-containers
    }
    
    if $images or $all {
        cleanup-images
    }
    
    if $volumes or $all {
        cleanup-volumes
    }
    
    if $all {
        docker system prune --force
        log success "✅ System cleanup completed"
    }
}

def cleanup-containers [] {
    log info "Cleaning up containers..."
    
    # Stop all polyglot containers
    try {
        let containers = docker ps -a --filter "name=*-dev" --format "{{.Names}}" | lines
        if ($containers | length) > 0 {
            docker stop ...$containers
            docker rm ...$containers
            log success $"✅ Removed ($containers | length) containers"
        } else {
            log info "No containers to clean up"
        }
    } catch { |e|
        log error $"❌ Failed to cleanup containers: ($e.msg)"
    }
}

def cleanup-images [] {
    log info "Cleaning up images..."
    
    try {
        let images = docker images --filter "reference=*-app" --format "{{.Repository}}:{{.Tag}}" | lines
        if ($images | length) > 0 {
            docker rmi ...$images
            log success $"✅ Removed ($images | length) images"
        } else {
            log info "No images to clean up"
        }
    } catch { |e|
        log error $"❌ Failed to cleanup images: ($e.msg)"
    }
}

def cleanup-volumes [] {
    log info "Cleaning up volumes..."
    
    try {
        docker volume prune --force
        log success "✅ Cleaned up unused volumes"
    } catch { |e|
        log error $"❌ Failed to cleanup volumes: ($e.msg)"
    }
}