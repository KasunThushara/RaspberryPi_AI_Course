# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install Python dependencies
#RUN pip install --no-cache-dir streamlit tensorflow pillow
RUN pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
