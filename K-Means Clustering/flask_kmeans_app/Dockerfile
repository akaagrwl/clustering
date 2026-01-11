# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the app
CMD ["python", "app.py"]
