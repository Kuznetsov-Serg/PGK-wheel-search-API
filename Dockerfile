ARG BASE_IMAGE=nexus.pgk.ru/custom-images/python:3.10-oracle-poetry-1.2.0

FROM ${BASE_IMAGE} as requirements-stage

WORKDIR /tmp

# RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM ${BASE_IMAGE}

WORKDIR /code

RUN echo "" > /etc/apt/sources.list && echo "" > /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        unzip \
        gcc \
        libsasl2-dev \
        python-dev-is-python3 \
        libaio1 \
        libldap2-dev \
#        libssl-dev \
    # Cleaning up temporary files and packages
    && apt-get autoremove -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install --no-cache-dir --upgrade PyYAML
RUN pip install --no-cache-dir --upgrade python-ldap


COPY ./config /code/config
COPY ./deploy/gunicorn_conf.py deploy/prestart* /code/
COPY ./deploy/entrypoint.sh deploy/start.sh /
COPY ./app /code/app

CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn_conf.py"]
