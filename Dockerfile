FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by pypdfium2 and other potential libraries
# libgl1-mesa-glx: required by some image processing libraries
# libgirepository1.0-dev, libcairo2-dev, libpangocairo-1.0-0, libgdk-pixbuf2.0-dev, shared-mime-info:
#   often required for PDF processing or UI-related Python packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgirepository1.0-dev \
    libcairo2-dev \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-dev \
    shared-mime-info \
    # Clean up apt caches to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Prevents pip from storing cached downloads, reducing image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Command to run your ETL script (this will be overridden by docker-compose's 'command')
# CMD ["python", "main_etl.py"]
