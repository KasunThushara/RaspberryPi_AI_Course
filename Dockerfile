# Use a base image with Python 3.11 installed
FROM arm64v8/python:3.11-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the contents of your app to the working directory
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir streamlit tensorflow ultralytics  # or tensorflow-lite if available

# Expose the default Streamlit port
EXPOSE 8501

# Specify the command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
