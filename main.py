import uvicorn

from app.settings import PARSED_CONFIG


def start_uvicorn():
    # uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run(
        "app.main:app",
        host=PARSED_CONFIG.server.host,
        port=PARSED_CONFIG.server.port,
        reload=True,
        workers=PARSED_CONFIG.server.workers,
    )


if __name__ == "__main__":
    start_uvicorn()
