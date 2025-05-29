FROM python:3.12-slim

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your application code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run your app
CMD ["python", "app.py"]
EXPOSE 5000
CMD gunicorn app:app --bind 0.0.0.0:$PORT
