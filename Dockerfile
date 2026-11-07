# syntax=docker/dockerfile:1.7
#
# fitdistcp dev environment — single source of truth for all toolchains.
# Host requires only Docker (or Podman); no Rust/R/Python installs needed.
#
# Build:  docker compose build
# Shell:  docker compose run --rm dev bash
#
# Stage 1: Rust builder (kept separate so the final image can reuse caches).
FROM ubuntu:26.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin

# ---------------------------------------------------------------------------
# System build tools + R (CRAN apt repo) + Python
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        cmake \
        git \
        curl \
        wget \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        libssl-dev \
        libcurl4-openssl-dev \
        libxml2-dev \
        libfontconfig1-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# R via the CRAN apt repository (pinned to Ubuntu 26.04 "resolute")
# ---------------------------------------------------------------------------
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | gpg --dearmor -o /usr/share/keyrings/cran.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/ubuntu resolute-cran40/" \
        > /etc/apt/sources.list.d/cran.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages needed for rextendr-based development and testing.
# Use Posit Public Package Manager (P3M) to get prebuilt Ubuntu binaries with
# full dependency resolution — far faster and more reliable than source builds.
# P3M only serves binaries when the HTTPUserAgent identifies the R version and
# platform, so we set it explicitly (otherwise it falls back to source builds).
RUN R -e "options(HTTPUserAgent = sprintf('R/%s R (%s)', getRversion(), paste(getRversion(), R.version['platform'], R.version['arch'], R.version['os']))); options(repos = c(P3M = 'https://packagemanager.posit.co/cran/__linux__/resolute/latest'), Ncpus = max(1L, parallel::detectCores())); install.packages(c('rextendr','devtools','testthat','roxygen2','usethis'))"

# ---------------------------------------------------------------------------
# Rust toolchain via rustup (stable, pinned by rust-toolchain.toml at build time)
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
            --component rustfmt --component clippy \
    && rustup toolchain install 1.85.0 --profile minimal \
            --component rustfmt --component clippy \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# ---------------------------------------------------------------------------
# Python tooling: maturin, pytest, numpy, cibuildwheel
# ---------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        maturin \
        pytest \
        numpy \
        cibuildwheel

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# Pre-fetch dependencies so rebuilds are fast (best-effort; ignores failures
# when the workspace is still being scaffolded).
RUN cargo fetch || true

CMD ["bash"]
