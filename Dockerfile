FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.save \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
