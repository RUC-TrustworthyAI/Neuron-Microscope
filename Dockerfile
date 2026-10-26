ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ARG DEBIAN_MIRROR=https://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git jq ripgrep \
    && rm -rf /var/lib/apt/lists/*

ARG DSH_VERSION=0.2.1rc2
ARG PYPI_INDEX=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --index-url "${PYPI_INDEX}" \
    "deepseek-harness-sdk==${DSH_VERSION}" \
    "deepseek-harness-runtime-bin==${DSH_VERSION}"
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
