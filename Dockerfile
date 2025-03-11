FROM python:3.10

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for logs
RUN mkdir -p /app/trading_logs /shared_logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SHARED_LOG_DIR=/shared_logs
ENV INSTANCE_LOG_DIR=/app/trading_logs

# Command to run the script
CMD ["python", "tutorials/FinRL_PortfolioAllocation_Explainable_DRL/scripts/a2c_paper_trading.py"]
