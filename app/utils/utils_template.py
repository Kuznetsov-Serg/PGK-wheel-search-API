import urllib
from typing import Any

from fastapi import Request

from app.settings import PARSED_CONFIG


def my_url_for(request: Request, name: str, **path_params: Any) -> str:
    url = request.url_for(name, **path_params)
    parsed = list(urllib.parse.urlparse(url))
    # parsed[1] = 'my_domain.com'
    parsed[0] = "https" if PARSED_CONFIG.IS_HTTPS and parsed[1] != "127.0.0.1:8000" else parsed[0]
    return urllib.parse.urlunparse(parsed)
