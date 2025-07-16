import datetime
import enum

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import BigInteger, Column, Date, String, UniqueConstraint, Text
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Session

from app.core.database import Base
from app.settings import PARSED_CONFIG


class MyLogTypeEnum(enum.Enum):
    START = "start"
    FINISH = "finish"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


LOG_TYPES = {
    MyLogTypeEnum.START: "Стартовал",
    MyLogTypeEnum.FINISH: "Завершено",
    MyLogTypeEnum.DEBUG: "debug",
    MyLogTypeEnum.INFO: "info",
    MyLogTypeEnum.WARNING: "warning",
    MyLogTypeEnum.ERROR: "error",
}


class Log(Base):
    __tablename__ = "log"
    __table_args__ = (
        UniqueConstraint("parent_id", "parent_name"),
        {"comment": "Таблица истории обработки"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    date = Column(Date, default=datetime.date.today, comment="Дата создания")
    parent_id = Column(BigInteger, comment="ID родительского объекта")
    parent_name = Column(String(20), comment="Наименование типа объекта/таблицы")
    type = Column(ENUM(MyLogTypeEnum), default="start", comment="Статус")
    msg = Column(Text, comment="Сообщение")

    @classmethod
    def write(cls, msg_type, msg, *args):
        cls.objects.create(type=msg_type, msg=msg + " " + " ".join([str(s) for s in args]))


class LogSchema(BaseModel):
    id: int
    date: datetime.date
    parent_id: int
    parent_name: str
    type: MyLogTypeEnum
    msg: str

    class Config:
        orm_mode = True


class LogDB:
    def __init__(self, db: Session, username: str = None) -> None:
        self.db = db
        self.id = 0
        self.username = username

    def get_list(self, filter_by: str = "", skip: int = 0, limit: int = 100):
        return (
            self.db.query(Log)
            .filter(Log.parent_name.ilike(f"%{filter_by}%"))
            .order_by(Log.id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return (
            db.query(Log).offset(skip).limit(limit).all()
            if parent_name == ""
            else db.query(models.Log).offset(skip).limit(limit).all()
        )

    def get(self, log_id: int = None, parent_id: int = None, parent_name: str = None):
        if not log_id and not (parent_id and parent_name):
            if self.id:
                log_id = self.id
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"For create Log (log_id={log_id}) need not empty "
                    f"parent_id ({parent_id}) and parent_name ({parent_name})",
                )
        if log_id:
            result = self.db.query(Log).filter(Log.id == log_id).first()
            if result is None:
                raise HTTPException(status_code=404, detail=f"Log with ID={log_id} not found")
        else:
            result = (
                self.db.query(Log).filter(Log.parent_id == parent_id).filter(Log.parent_name == parent_name).first()
            )
        if result:
            self.id = result.id
        return result

    def add(self, *args, **kwargs):
        return self._add(*args, **kwargs, is_append=True)

    def put(self, *args, **kwargs):
        return self._add(*args, **kwargs, is_append=False)

    def _add(
        self,
        log_id: int = None,
        parent_id: int = None,
        parent_name: str = None,
        type_log: MyLogTypeEnum = None,
        msg: str = "",
        is_append: bool = True,
        is_with_time: bool = True,
        username: str = "",
    ):
        if not username:
            username = self.username if self.username else PARSED_CONFIG.username
        if is_with_time and msg:
            msg = f'{str(datetime.datetime.today()).split(".", 2)[0]} ({username}) - {msg}'
        db_log = self.get(log_id, parent_id, parent_name)
        if not db_log:
            if not parent_id or not parent_name:
                raise HTTPException(
                    status_code=404,
                    detail=f"For create Log (log_id={log_id}) need not empty "
                    f"parent_id ({parent_id}) and parent_name ({parent_name})",
                )
            type_log = type_log if type_log else MyLogTypeEnum.INFO
            db_log = Log(parent_id=parent_id, parent_name=parent_name[:20], type=type_log, msg=msg)
            self.db.add(db_log)
            self.db.commit()
            if parent_id < 0:  # without an entity (ID<0), let's take a fake ID in the database
                self.db.query(Log).filter(Log.id == db_log.id).update({"parent_id": db_log.id})
                self.db.commit()
        else:
            log_update_dict = dict()
            log_update_dict["type"] = type_log if type_log else db_log.type
            log_update_dict["msg"] = f"{db_log.msg}\n{msg}" if is_append else msg

            self.db.query(Log).filter(Log.id == db_log.id).update(log_update_dict)
            self.db.commit()

        self.id = db_log.id
        return self.get()
