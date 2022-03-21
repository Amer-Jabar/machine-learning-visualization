# Section 1- Base Image
FROM python:3.8-slim

# Section 2- Python Interpreter Flags
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Section 3- Compiler and OS libraries
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Section 4- Project libraries and User Creation
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Section 5- Copy everything
WORKDIR /app

COPY . .

# Section 6- Exposing Port 8080
ENV PORT 8080
EXPOSE 8080

# Most Important Part. Setting Django Settings
ENV DJANGO_SETTINGS_MODULE machine_learning_visualization.settings

# Section 7- Docker Run Checks and Configurations
CMD python -m django runserver 0.0.0.0:8080