FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
    && apt-get -qqq install --no-install-recommends -y pkg-config gcc g++ python3.8-dev python3-pip libpython3.8-dev git \
    && apt-get clean \
    && rm -rf /var/lib/apt

WORKDIR /app/

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY . .

EXPOSE 8080
CMD flask --app whisper_web.app run -p 8080 -h 0.0.0.0
