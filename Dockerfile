# Use a base image with Python 3.11 installed
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of your app to the working directory
COPY . /app

# Install any dependencies your app needs (make sure Streamlit is in requirements.txt)
RUN pip install -r requirements.txt

# If you don't have requirements.txt, uncomment the line below to install Streamlit directly
# RUN pip install streamlit

# Expose the default Streamlit port
EXPOSE 8501

# Specify the command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
