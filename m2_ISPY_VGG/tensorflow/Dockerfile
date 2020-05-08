FROM python:3.7-slim-buster

LABEL maintainer="@jeadie"

RUN apt-get -y update

# Install packages to allow apt to use a repository over HTTPS
RUN apt-get -y install \
    curl \
    gnupg2

# Download GCSFuse and then upgrade remaining packages.
RUN export LSB_RELEASE=bionic

RUN export GCSFUSE_REPO="gcsfuse-bionic" && \
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get -y update && \
    apt-get -y install gcsfuse && \ 
    apt-get upgrade -y && \
    apt-get -y autoremove

# Get Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

ENV PYTHONUNBUFFERED=1
COPY *.py ./
