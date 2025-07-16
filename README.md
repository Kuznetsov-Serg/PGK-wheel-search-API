# Wheel search - API
### (Возмещение ущерба MVP)

Search for wheelsets (a project for the security service<br>
<i>(Возмещение убытков от хищений (подмен), повреждений колесных пар + Дашборд возмещения убытков)</i><br>


## Features:

#### [Confluence docs](https://conf.pgk.ru/pages/viewpage.action?pageId=275128721)


[![pipeline status](https://gitlab.pgk.ru/poc/wheel-search-api/-/pipelines)](https://gitlab.pgk.ru/poc/wheel-search-api/-/pipelines)<br>
[![commits](https://gitlab.pgkweb.ru/poc/-/commits)](https://gitlab.pgk.ru/poc/wheel-search-api/-/commits/develop?ref_type=heads)<br>

## Установка и запуск проекта

Перейти в корень проекта: /wheel-search-api/

```bash
pip install poetry==1.2.1 # Установка poetry
poetry self update --preview  # актуализируем версию poetry
poetry lock # создать poetry.lock
poetry install # Установка зависимостей из файла pyproject.toml
poetry run server # Локальный запуск uvicorn
```
```bash
если poetry lock долго отрабатывает, можно зависимости установить:
poetry export -f requirements.txt > requirements.txt
pyth

#### pre-commit hooks:

После клонирования репозитория обязательно установить pre-commit hooks.

Через poetry

```bash
poetry add -D pre-commit
poetry run pre-commit install
```

Глобальное окружение

```bash
pip install pre-commit
pre-commit install
```


## Requirements:


 - Debian / Ubuntu / Windows Subsystem for Linux
 - Python 3.10
 - Poetry

## Environments
**_(necessary environment variables)_**

Description of the project:
- PROJECT_NAME="wheel-search-api"
- PROJECT_VERSION=2.0
- PROJECT_ENVIRONMENT=dev

Authorization and authentication parameters
- IS_AUTHORISE_BY_TOKEN=True
- IS_AUTHORISE_BY_ROLE=True
- LDAP_SERVER=10.144.52.13

Token
- JWT_SECRET=*****************************
- JWT_ALGORITHM=HS512
- JWT_ACCESS_TOKEN_DAYS=2

Restrictions on requests (if necessary)
- FAIL_LOGIN_TRY=3 
- FAIL_LOGIN_PERIOD_BLOCK=10

The number of items in the database query lists
- CHUNK_COUNT_PER_SQL=500

Addresses and credits
- API_PREFIX=/api (or empty str)
- IS_HTTPS=True

Logging of CRUD
- IS_LOG_CRUD_CHANGE=True
- IS_LOG_CRUD_ALL=False

Server
- SERVER_HOST=127.0.0.1
- SERVER_PORT=9000
- SERVER_WORKERS=8

Error control
- IS_SENTRY=False
- SENTRY_DSN=https://43f1cc1dfd7f4f4ab089ed9205e1148a@sentry-dev.pgk.ru/8

Postgres DataBase
- DB_POSTGRES_DSN=postgresql://UserName:Password@127.0.0.1:5432/wheel-search-api
- DB_POSTGRES_HOST=127.0.0.1
- DB_POSTGRES_PORT=5432
- DB_POSTGRES_USER=UserName
- DB_POSTGRES_PASS=Password
- DB_POSTGRES_NAME=wheel-search-api

Oracle Database (showcase)
- DB_ORACLE_DSN=oracle+cx_oracle://UserName:Password@komandor-db:1521/orcl
- DB_ORACLE_HOST=komandor-db
- DB_ORACLE_PORT=1521
- DB_ORACLE_USER=UserName
- DB_ORACLE_PASS=Password
- DB_ORACLE_NAME=orcl
