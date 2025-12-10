#!/bin/bash

# ruv-swarm Backend Startup Script

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ruv-swarm backend...${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}Created .env file. Please update it with your configuration.${NC}"
    else
        echo -e "${RED}Error: .env.example not found. Cannot create .env file.${NC}"
        exit 1
    fi
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Validate required environment variables
REQUIRED_VARS=("JWT_SECRET" "DATABASE_URL")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}Error: Required environment variable $var is not set.${NC}"
        exit 1
    fi
done

# Check JWT_SECRET strength in production
if [ "$NODE_ENV" = "production" ] && [ ${#JWT_SECRET} -lt 32 ]; then
    echo -e "${RED}Error: JWT_SECRET must be at least 32 characters in production.${NC}"
    exit 1
fi

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo -e "${GREEN}Created logs directory${NC}"
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Run database migrations/initialization
echo -e "${GREEN}Initializing database...${NC}"

# Start the server based on environment
if [ "$NODE_ENV" = "production" ]; then
    echo -e "${GREEN}Starting server in production mode...${NC}"
    npm start
else
    echo -e "${GREEN}Starting server in development mode...${NC}"
    npm run dev
fi