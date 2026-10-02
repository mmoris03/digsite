# The digsite command line, ready to crawl, index, search, answer and serve.
# The corpus lives in /app/data and the embedding model in the Hugging Face
# cache: both are meant to be volumes, so neither is rebuilt with the image.

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache

WORKDIR /app

# The CPU build of PyTorch: the default one brings gigabytes of CUDA libraries
# that this image never uses.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.9"

# Dependencies before the code, so that changing the code reuses this layer.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

USER app
EXPOSE 8000
ENTRYPOINT ["digsite"]
CMD ["serve", "--host", "0.0.0.0"]
