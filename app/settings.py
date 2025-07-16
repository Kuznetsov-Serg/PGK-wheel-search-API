import datetime
import os
from enum import Enum
from functools import lru_cache
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, PositiveInt, StrictStr

from app.utils.utils import merge, read_yaml

PROJ_ROOT = Path(__file__).parent.parent
config_env_var = "DVU_CONFIG_PATH"
DEFAULT_PATH = PROJ_ROOT / "config" / "config.yaml"
DVU_CONFIG_PATH = PROJ_ROOT / "config" / "docker.yaml"
STATIC_PATH = PROJ_ROOT / "static"
TEMPLATES_PATH = PROJ_ROOT / "templates"
# TEMPLATES_PATH_VIEWS = TEMPLATES_PATH / 'views'
ENV_PATH = Path(os.environ.get(config_env_var) or "")

COOKIE_EXPIRATION_TIME = datetime.datetime.now() + datetime.timedelta(days=1000)
COOKIE_EXPIRATION_DATE = COOKIE_EXPIRATION_TIME.strftime("%a, %d %b %Y %H:%M:%S GMT")

# MAP_DIR = PROJ_ROOT / "assembled_map"
# MAP_NAME = 'map.json'
# MAP_PATH = MAP_DIR / MAP_NAME

dotenv_path = Path(__file__).parent.parent.joinpath(".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

LoggingConfig = dict[str, Any]


def get_variable(name: str, default_value: bool | None = None) -> bool:
    true_ = ("true", "1", "t")  # Add more entries if you want, like: `y`, `yes`, `on`, ...
    false_ = ("false", "0", "f")  # Add more entries if you want, like: `n`, `no`, `off`, ...
    value: str | None = os.getenv(name, None)
    if value is None:
        if default_value is None:
            raise ValueError(f"Variable `{name}` not set!")
        else:
            value = str(default_value)
    if value.lower() not in true_ + false_:
        raise ValueError(f"Invalid value `{value}` for variable `{name}`")
    return value in true_


class EmptyCustomConfig(Exception):
    def __init__(self, path: Path):
        self.path = path

    def __str__(self) -> str:
        return f"Config file {self.path} is empty"


# class SMTPClient(ConnectionConfig):
#     MAIL_USERNAME: StrictStr
#     MAIL_PASSWORD: StrictStr
#     MAIL_PORT: PositiveInt = 465
#     MAIL_SERVER: str
#     MAIL_TLS: bool = False
#     MAIL_SSL: bool = True
#     MAIL_DEBUG: conint(gt=-1, lt=2) = 0  # type: ignore
#     MAIL_FROM: EmailStr
#     MAIL_FROM_NAME: Optional[str] = None
#     SUPPRESS_SEND: conint(gt=-1, lt=2) = 0  # type: ignore
#     USE_CREDENTIALS: bool = True
#     VALIDATE_CERTS: bool = True
#


class ServerConfiguration(BaseModel):
    host: StrictStr
    port: PositiveInt
    workers: int
    root_url: Optional[StrictStr] = None


class AdAuthorizeConfiguration(BaseModel):
    is_enable: bool = False
    user: str = ""
    password: str = ""
    user_group: str = ""
    admin_group: str = ""
    security_admin_group: str = ""


class DependencyConfiguration(BaseModel):
    dsn: StrictStr


class DatabaseConfiguration(DependencyConfiguration):
    db_host: str
    db_port: str
    db_user: str
    db_pass: str
    db_name: str
    options: Optional[str] = ""


class CeleryConfiguration(DependencyConfiguration):
    tasks: StrictStr


class JWTConfiguration(BaseModel):
    jwt_secret: str
    jwt_algorithm: str = "RS256"
    jwt_access_token_days: int = 2


class RateLimit(BaseModel):
    is_enable: bool = False
    second: int = 10
    minute: int = 100


class Configuration(BaseModel):
    project_name: StrictStr
    project_version: str
    project_environment: str
    backend_cors_origins: list[AnyHttpUrl] = []
    is_https: bool = True
    api_prefix: str = ""
    root_path: str = ""

    is_authorise_by_token: bool = False
    is_authorise_by_role: bool = False

    gzip_minimums_size: int = 500
    is_sentry: bool = False
    sentry_dsn: str = ""

    server: ServerConfiguration
    logging: LoggingConfig

    is_log_crud_change: bool = False
    is_log_crud_all: bool = False

    database: DatabaseConfiguration
    database_ora: DatabaseConfiguration
    # smtp_client: SMTPClient
    # celery: CeleryConfiguration
    # flower: DependencyConfiguration
    ad_authorize: AdAuthorizeConfiguration
    jwt: JWTConfiguration
    rate_limit: RateLimit
    ldap_server: StrictStr = "10.144.52.13"
    fail_login_try: int = 3
    fail_login_period_block: int = 10
    username: str = None
    user_access: str = None
    tables_tracked_for_authorization: list[str] = None


class SourceType(str, Enum):
    SAP = "SAP"
    ASU_VRK = "ASU VRK"
    VAREKS = "Vareks"
    DAMAGE_LIST = "List of damaged parts"


@lru_cache
def load_configuration(path: Path = "") -> Configuration:
    arg_path = Path(path)
    default_config = read_yaml(DEFAULT_PATH)

    custom_config_path = (arg_path.is_file() and arg_path) or (ENV_PATH.is_file() and ENV_PATH)
    if custom_config_path:
        custom_config = read_yaml(custom_config_path)

        if not custom_config:
            raise EmptyCustomConfig(path=custom_config_path)
        config_data = merge(default_config, custom_config)
    else:
        config_data = default_config

    config_data["tables_tracked_for_authorization"] = get_tables_tracked_for_authorization()
    # remove None
    config_data = {
        key: {key1: val1 for key1, val1 in val.items() if val1 != "None"} if isinstance(val, dict) else val
        for key, val in config_data.items()
        if val != "None"
    }

    return Configuration(**config_data)


def setup_logging(logging_config: LoggingConfig):
    dictConfig(logging_config)


def dump_config(config: Configuration) -> str:
    return config.json(indent=2, sort_keys=True)


def get_tables_tracked_for_authorization() -> list:
    return get_tables_admin_tracked_for_authorization() + get_tables_user_tracked_for_authorization()


def get_tables_admin_tracked_for_authorization() -> list:
    return ["role", "role_item", "user", "user_role", "user_history"]


def get_tables_user_tracked_for_authorization() -> list:
    # tables = []
    # for key, data in inspect.getmembers(models, inspect.isclass):
    #     if isinstance(data, DeclarativeMeta):
    #         if 'metadata' in data.__dict__ and 'tables' in data.metadata.__dict__:
    #             tables = list(data.metadata.tables)
    #             break

    tables = [
        "work_filter",
        "work_filter_by_storage",
        "work_cost",
        "steal",
        "check_filter",
        "wheel_set",
        "wheel_set_filter",
        "wheel_set_cost",
        "mounting_type",
        "mounting_type_map",
        "storage",
    ]
    return tables


def get_all_access_false() -> dict:
    return {table: "0" * 6 for table in get_tables_tracked_for_authorization()}


def get_all_access_true() -> dict:
    return {table: "1" * 6 for table in get_tables_tracked_for_authorization()}


def get_restriction_for_user() -> dict:
    result = get_all_access_false()
    result["role_item"] = "010000"
    result["user_role"] = "010000"
    result["user_history"] = "010000"
    return result


def get_restriction_for_admin() -> dict:
    result = get_all_access_false()
    result["role"] = "010001"
    result["role_item"] = "010001"
    result["user"] = "010001"
    result["user_role"] = "010001"
    result["user_history"] = "010001"
    result["wheel_set"] = "010001"
    result["wheel_set_filter"] = "010001"
    result["wheel_set_cost"] = "010001"
    result["mounting_type"] = "010001"
    result["mounting_type_map"] = "010001"
    result["work_filter"] = "010001"
    result["work_filter_by_storage"] = "010001"
    result["work_cost"] = "010001"
    result["steal"] = "010001"
    result["check_filter"] = "010001"
    return result


def get_restriction_for_security_admin() -> dict:
    result = get_all_access_true()
    result["user_history"] = "010001"
    result["wheel_set"] = "010001"
    result["wheel_set_filter"] = "010001"
    result["wheel_set_cost"] = "010001"
    result["mounting_type"] = "010001"
    result["mounting_type_map"] = "010001"
    result["work_filter"] = "010001"
    result["work_filter_by_storage"] = "010001"
    result["work_cost"] = "010001"
    result["steal"] = "010001"
    result["check_filter"] = "010001"
    return result


PARSED_CONFIG = load_configuration()

# COOKIE_EXPIRATION_TIME = datetime.datetime.now() + datetime.timedelta(days=1000)
# COOKIE_EXPIRATION_DATE = COOKIE_EXPIRATION_TIME.strftime("%a, %d %b %Y %H:%M:%S GMT")
# CURRENT_YEAR_INT = datetime.datetime.today().year
EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
