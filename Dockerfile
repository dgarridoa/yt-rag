FROM python:3.11

# Spark dependency
RUN apt update && apt -y install openjdk-17-jdk

# Install Databricks CLI
RUN curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Install Poetry
RUN pip install poetry==1.8.3

# Set working directory
WORKDIR /app

# Copy only files necessary for installing dependencies
COPY pyproject.toml *.lock .

# Install python dependencies
RUN poetry install --with dev,test --no-root

# Copy the rest of the application
COPY . .
