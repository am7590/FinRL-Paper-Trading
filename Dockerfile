FROM python:3.10.15

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Initialize and update submodules (for finrl)
RUN git submodule init && git submodule update --recursive

# Copy the entire project directory into the container
COPY . /app

# Install the local finrl package
RUN pip install -e ./finrl

# Default command to run the main paper trading script
CMD ["python", "tutorials/FinRL_PaperTrading_Demo/scripts/paper_trading.py"]
