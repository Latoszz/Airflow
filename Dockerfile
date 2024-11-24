# Use the official Airflow image as the base image
FROM apache/airflow:2.10.3

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
