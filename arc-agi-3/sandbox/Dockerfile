FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG NODE_VERSION=22.22.2
ARG CLAUDE_CODE_VERSION=2.1.216

RUN apt-get update -q \
    && apt-get install -y -q --no-install-recommends \
        git ca-certificates curl xz-utils procps bash \
    && arch="$(dpkg --print-architecture)" \
    && case "$arch" in amd64) node_arch=x64 ;; arm64) node_arch=arm64 ;; \
        *) echo "unsupported arch: $arch" >&2; exit 1 ;; esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
        | tar -xJ --no-same-owner -C /usr/local --strip-components=1 \
    && npm install -g --no-fund --no-audit "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && pip install --no-cache-dir arc-agi==0.9.8 arcengine==0.9.3 numpy scipy pydantic rich httpx \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /root/.npm /tmp/* \
    && node --version && claude --version \
    && python3 -c "import arc_agi, arcengine, numpy, scipy; print('py deps ok')"

ENV HOME=/tmp
WORKDIR /opt/arc/src
CMD ["/bin/bash"]
