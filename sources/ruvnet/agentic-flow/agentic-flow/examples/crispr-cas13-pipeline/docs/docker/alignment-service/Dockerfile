# Alignment Service with Bowtie2 and Bioconda
FROM continuumio/miniconda3:latest

LABEL maintainer="CRISPR-Cas13 Team"
LABEL description="Alignment service with Bowtie2, Samtools, FastQC"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure conda channels
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Install bioinformatics tools
RUN conda install -y \
    python=3.11 \
    bowtie2=2.5.1 \
    samtools=1.17 \
    fastqc=0.12.1 \
    multiqc=1.16 \
    && conda clean -afy

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY alignment_worker.py /app/
COPY utils.py /app/

# Create non-root user
RUN useradd -u 1000 -m -s /bin/bash worker && \
    chown -R worker:worker /app

USER worker

# Verify installations
RUN bowtie2 --version && \
    samtools --version && \
    fastqc --version && \
    python --version

EXPOSE 9090

CMD ["python", "alignment_worker.py"]
