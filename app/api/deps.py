from collections.abc import Generator

from app.core.database import ConnectionLocal, EnginePostresql, SessionLocal, EngineOracle, SessionLocalOra


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def get_cursor() -> Generator:
    try:
        connection = ConnectionLocal()
        cursor = connection.cursor()
        yield cursor
    finally:
        # Закрываем соединение
        cursor.close()
        connection.close()


def get_engine() -> Generator:
    try:
        engine = EnginePostresql
        yield engine
    finally:
        # Закрываем соединение
        print("close get_engine")


def get_db_ora() -> Generator:
    try:
        db = SessionLocalOra()
        yield db
    finally:
        db.close()


def get_engine_ora() -> Generator:
    try:
        engine = EngineOracle
        yield engine
    finally:
        # Закрываем соединение
        print("clos get_ora_engine")
