FROM python:3.10-slim

# Set working directory
RUN mkdir /app

# Copy the rest of your app code
COPY app /app

# Copy dependency file 
COPY requirements.txt /app

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Launch the Streamlit app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--browser.gatherUsageStats=false"]
