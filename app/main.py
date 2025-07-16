import time

from fastapi import Depends, FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from ratelimit import RateLimitMiddleware, Rule

# from ratelimit.auths.ip import client_ip
from ratelimit.backends.simple import MemoryBackend

from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Scope

from app.api.router import routers_lst, router_test
from app.auth.token_ldap import check_token
from app.auth.router import auth_router
from app.auth.router_crud import routers_lst as routers_auth_lst
from app.core import models

# from app.utils.docs import use_route_names_as_operation_ids
from app.core.database import EnginePostresql
from app.settings import Configuration, load_configuration, dump_config
from app.utils.exceptions import api_error_responses, http_exception_handler, validation_exception_handler
from app.utils.responses import WrappedResponse
from app.utils.sentry import init_sentry


# from starlette_validation_uploadfile import ValidateUploadFileMiddleware


class CheckApiKey(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # print('configuration=', configuration)
        print("request=", request)
        if configuration.is_authorise_by_token:
            token = request.headers.get("authorization", None)
            print(f"CheckApiKey token={token}")
            try:
                token = check_token(token[7:])
            except:
                print("Wrong token")
                return JSONResponse(
                    status_code=401,
                    content={"error_code": "server_problem", "errorMessage": "Not authenticated"},
                )
                # response = Response(status_code=401, content=b'{"error_code": "server_problem",
                # "errorMessage": "Not authenticated"}')
                # response = await call_next(request)
                # return response
            else:
                request.headers["username"] = token.get("sub", "NoAuthorised")
        response = await call_next(request)

        return response


async def auth_function(scope: Scope) -> tuple[str, str]:
    """
    Resolve the user's unique identifier and the user's group from ASGI SCOPE.

    If there is no user information, it should raise `EmptyInformation`.
    If there is no group information, it should return "default".
    """
    try:
        cookie = [el[1].decode() for el in scope["headers"] if el[0] == b"cookie"]
        user_id = "".join([el[8:] for el in cookie if el.startswith("user_id=")])
        # client = str(scope.get('client'))
        real_ip = ",".join([el[1].decode() for el in scope["headers"] if el[0] == b"x-original-forwarded-for"])
        user_id = user_id if user_id else real_ip
    except:
        user_id = "default"
    group_name = "default"
    # print(f"user_id={user_id}")
    # print(f"user_id={user_id}\nscope={scope}")
    return user_id, group_name


def create_app(config: Configuration) -> FastAPI:
    root_path = f"/v1/{config.project_name}" if config.project_environment == "dev" else ""
    application = FastAPI(
        title=config.project_name,
        description=f"Environment: {config.project_environment}",
        version=config.project_version,
        default_response_class=WrappedResponse,
        root_path=root_path,
        docs_url=config.api_prefix + "/docs",
    )

    application.add_middleware(
        CORSMiddleware,
        # allow_origins=[str(origin) for origin in config.BACKEND_CORS_ORIGINS],
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(
        GZipMiddleware,
        minimum_size=config.gzip_minimums_size,
    )
    # Косячит (несовместимо)
    # application.add_middleware(
    #     ValidateUploadFileMiddleware,
    #     app_path="/api/steal-load/",
    #     max_size=10000024,  # 1 Kbyte
    #     file_type=["image/png", "image/jpeg"]
    # )

    # Any incoming requests to http or ws will be redirected to the secure scheme instead (https or wss)
    # application.add_middleware(HTTPSRedirectMiddleware)

    # application.add_middleware(CheckApiKey)
    if config.is_sentry:
        init_sentry()
        application.add_middleware(SentryAsgiMiddleware)

    application.exception_handler(HTTPException)(http_exception_handler)
    application.exception_handler(RequestValidationError)(validation_exception_handler)
    application.state.config = config
    application.include_router(router_test)
    if config.is_authorise_by_token and config.is_authorise_by_role:
        application.include_router(auth_router)
        for router in routers_auth_lst:
            application.include_router(router, responses=api_error_responses)
    for router in routers_lst:
        application.include_router(router, responses=api_error_responses)

    # authenticate = auth_function if os.name == "nt" else client_ip

    if config.rate_limit.is_enable:
        application.add_middleware(
            RateLimitMiddleware,
            # authenticate=client_ip,
            authenticate=auth_function,
            backend=MemoryBackend(),
            config={
                # does not contain "-check" (not restrict test EndPoints)
                r"^((?!-check).)*$": [
                    Rule(second=config.rate_limit.second, minute=config.rate_limit.minute, group="default")
                ],
            },
        )

    # use_route_names_as_operation_ids(application)
    # application.mount(
    #     "/static", StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"), name="/static"
    # )

    return application


configuration = load_configuration()
# setup_logging(configuration.logging)
print(f"Loaded configuration\n {dump_config(configuration)}")
app = create_app(configuration)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next, token=Depends(check_token)):
    # async def add_process_time_header(request: Request, call_next, token: str = Depends(oauth2_scheme)):
    start_time = time.time()
    # token = await check_token_wrap(token)
    # print(f'!!! token={token}')
    response = await call_next(request)
    # print(f'!!!_!!! token={token}')
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


models.Base.metadata.create_all(bind=EnginePostresql)
# app = get_application()
