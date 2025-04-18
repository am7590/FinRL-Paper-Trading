# Rest of the Dockerfile remains unchanged
FROM python:3.10-slim

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --timeout 1000 --retries 3 --no-deps -r requirements.txt

RUN pip install --no-cache-dir --timeout 600 torch --index-url https://download.pytorch.org/whl/cpu

# Install Google API dependencies
RUN pip install --no-cache-dir \
    google-auth-oauthlib \
    google-auth-httplib2 \
    google-api-python-client \
    requests-oauthlib \
    google-auth

# Copy the rest of the application
COPY . .

# Create directories for logs and credentials
RUN mkdir -p /app/trading_logs/shared_logs /app/credentials

# Set permissions
RUN chmod -R 777 /app/trading_logs/shared_logs /app/credentials

# Set environment variables
ENV PYTHONPATH=/app
ENV SHARED_LOG_DIR=/shared_logs
ENV INSTANCE_LOG_DIR=/app/trading_logs
ENV GOOGLE_DOCS_CREDENTIALS_PATH=/app/credentials/service_account.json

# Create a volume for credentials
VOLUME ["/app/credentials"]

# Command to run the script
CMD ["python", "tutorials/FinRL_StockTrading_Fundamental/scripts/a2c_paper_trading.py"]
