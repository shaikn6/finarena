# syntax=docker/dockerfile:1
# Base:  docker build -t finarena .
# + LLM: docker build --build-arg EXTRAS='[llm]' --build-arg PRELOAD_LLM=1 -t finarena:llm .
#        (torch is installed from the CPU wheel index: the default wheel drags ~9 GB of CUDA libraries into a CPU container)
#
# Layer order is chosen so day-to-day source edits rebuild in about a minute (not a full reinstall): dependencies and the baked-in model weights
# depend only on pyproject.toml, and the application source is installed last with --no-deps.
FROM python:3.12-slim AS build
ARG EXTRAS=""
ARG PRELOAD_LLM=0
ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
WORKDIR /app
COPY pyproject.toml ./
# Dependencies only: install against an empty package stub so this layer is cached until pyproject.toml changes.
RUN mkdir -p src/finarena && touch src/finarena/__init__.py \
 && pip install --no-cache-dir --prefix=/install ".${EXTRAS}"
# Bake the base model into the image so startup never depends on the network.
RUN mkdir -p /opt/hf && if [ "$PRELOAD_LLM" = "1" ]; then \
      PYTHONPATH=/install/lib/python3.12/site-packages HF_HOME=/opt/hf \
      python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-0.5B-Instruct')"; fi
COPY src ./src
# The stub install left a build/ cache; setuptools would otherwise keep the stub's empty __init__.py as 'newer'.
RUN rm -rf build src/*.egg-info && pip install --no-cache-dir --prefix=/install --no-deps --force-reinstall .

FROM python:3.12-slim
ARG PRELOAD_LLM=0
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build /opt/hf /opt/hf
COPY artifacts ./artifacts
RUN chmod -R a+rX /app/artifacts /opt/hf  # host files may be 0600; the service runs as an unprivileged user
ENV FINARENA_ARTIFACT_DIR=/app/artifacts PYTHONUNBUFFERED=1 HF_HOME=/opt/hf HF_HUB_OFFLINE=${PRELOAD_LLM}
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "finarena.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
