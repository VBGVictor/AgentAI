FROM python:3.9-slim

# Install dependencies
RUN pip install --upgrade pip
RUN pip install tensorflow transformers

# Set the working directory
WORKDIR /app

# Copy your script into the container
COPY agent.py /app/agent.py

# Run the script
CMD ["python", "agent.py"]