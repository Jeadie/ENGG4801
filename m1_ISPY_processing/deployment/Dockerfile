FROM python:3.7-slim-buster


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

ENV PYTHONUNBUFFERED=1
COPY ISPY1_MetaData.csv .
COPY pipeline/*.py ./pipeline/
COPY main.py .
COPY pipeline/*.sh ./pipeline/
COPY pipeline/*.csv ./pipeline/

RUN mkdir output/
