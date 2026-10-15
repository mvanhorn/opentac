# The hosted OpenTac-CAD demo (Render). Same package as a local run, plus the
# [demo] extra for the Gemini design chat. Built from the repository root:
#   docker build -f deploy/Dockerfile .
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra demo --no-install-project
COPY opentac ./opentac
# The chat reads the same skill Claude Code loads; the landing gallery serves the example STLs.
COPY .claude/skills ./.claude/skills
COPY examples/stl ./examples/stl
RUN uv sync --locked --no-dev --extra demo
EXPOSE 10000
CMD ["uv", "run", "--no-sync", "opentac", "cad", "--host", "0.0.0.0", "--port", "10000", "--no-browser"]
