import sentry_sdk
from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.settings import PARSED_CONFIG


def http_exception_handler(request: Request, exc: HTTPException):
    if PARSED_CONFIG.is_sentry and exc.status_code != 400:
        sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": "server_problem", "errorMessage": exc.detail},
    )


def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error_code": "validation_error",
            "error_message": f"Неправильные параметры запроса: {exc}",
        },
    )


class ErrorMessage(BaseModel):
    error_code: str
    error_message: str


api_error_responses = {
    400: {"model": ErrorMessage},
    422: {"model": ErrorMessage},
}
