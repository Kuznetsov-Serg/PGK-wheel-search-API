import sentry_sdk

from app.settings import PARSED_CONFIG


def init_sentry():
    if PARSED_CONFIG.is_sentry:
        sample_rate = 1.0 if PARSED_CONFIG.project_environment != "prod" else 0.2
        sentry_sdk.init(
            PARSED_CONFIG.sentry_dsn,
            traces_sample_rate=sample_rate,
            environment=PARSED_CONFIG.project_environment,
            release=PARSED_CONFIG.project_version,
        )
