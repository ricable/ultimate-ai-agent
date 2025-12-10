FROM python:3.11-slim

# Build argument for port with default value
ARG PORT=8001
ENV PORT=${PORT}

WORKDIR /app

# Install system dependencies required for markitdown
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for files
RUN mkdir -p test_files markdown_results

# Expose the port from build argument
EXPOSE ${PORT}

# Command to run the application
# Feel free to change sample_supabase_agent to sample_postgres_agent
CMD ["sh", "-c", "uvicorn file_agent:app --host 0.0.0.0 --port ${PORT}"]
