# =============================
# Backend - Python Event Server
# =============================

FROM python:3.10-slim

# Ensure logging directory exists inside container
WORKDIR /app

# Install system libs
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose backend port
EXPOSE 5001

# Logs will be mounted from host
VOLUME ["/app/logs"]

# Run the unified server
CMD ["python", "server.py"]
