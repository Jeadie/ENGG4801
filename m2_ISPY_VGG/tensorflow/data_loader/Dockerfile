FROM python:3.7-slim-buster

LABEL maintainer="@jeadie"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

ENV PYTHONUNBUFFERED=1
COPY *.py ./
