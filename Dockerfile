FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV, MediaPipe and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Streamlit port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]