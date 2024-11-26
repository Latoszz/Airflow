# Use the official Airflow image as the base image
FROM apache/airflow:2.10.3-python3.11

# Switch to root to install system dependencies
USER root

# Install system dependencies required for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Set the working directory
WORKDIR /opt/airflow

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your DAGs, plugins, and configuration files into the container
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
COPY config /opt/airflow/config

# Continue with the default entrypoint
ENTRYPOINT ["/entrypoint"]