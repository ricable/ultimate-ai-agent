# Differential Expression Service with R/Bioconductor
FROM bioconductor/bioconductor_docker:RELEASE_3_18

LABEL maintainer="CRISPR-Cas13 Team"
LABEL description="Differential expression service with DESeq2 and edgeR"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "BiocManager::install(c( \
    'DESeq2', \
    'edgeR', \
    'Rsubread', \
    'ggplot2', \
    'pheatmap', \
    'EnhancedVolcano', \
    'limma', \
    'biomaRt', \
    'org.Hs.eg.db' \
    ), ask = FALSE)"

# Install Python dependencies for Kafka integration
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy R scripts
COPY diff_expr_worker.R /app/
COPY deseq2_analysis.R /app/
COPY edger_analysis.R /app/

# Copy Python orchestrator
COPY orchestrator.py /app/

# Create non-root user
RUN useradd -u 1000 -m -s /bin/bash worker && \
    chown -R worker:worker /app

USER worker

EXPOSE 9090

CMD ["python3", "orchestrator.py"]
