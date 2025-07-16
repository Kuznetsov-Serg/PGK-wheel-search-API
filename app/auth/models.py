import datetime
import enum
import inspect
import sys
from itertools import zip_longest

from sqlalchemy import (
    VARCHAR,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, DeclarativeMeta, validates

from app.core.database import Base
from app.settings import get_tables_tracked_for_authorization


class User(Base):
    __tablename__ = "user"
    __table_args__ = {"comment": "Пользователи"}

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="ID")
    username = Column(String, unique=True, index=True, comment="Имя пользователя")
    email = Column(String, unique=True, index=True, comment="Email")
    hashed_password = Column(String, nullable=True, comment="Закодированный пароль")
    is_active = Column(Boolean, default=True, comment="Активный (да/нет)")

    user_role = relationship("UserRole", cascade="all,delete", backref="user_role")
    # user_history = relationship("UserHistory", cascade="all,delete", backref="user_history")
    # user_login_history = relationship("UserLoginHistory", cascade="all,delete", backref="user_login_history")

    @property
    def user_role_list(self) -> list:
        return [el.id for el in self.user_role] if self.user_role else None

    # @property
    # def access_dict(self):
    #     result = dict()
    #     if self.user_role:
    #         for role in self.user_role:
    #             if role.role.role_item:
    #                 for el in role.role.role_item:
    #                     if el.item in result:
    #                         result[el.item] = "".join(
    #                             [
    #                                 (el1 if el1 == "1" else el2)
    #                                 for el1, el2 in zip_longest(result[el.item], el.access, fillvalue=0)
    #                             ]
    #                         )
    #                     else:
    #                         result[el.item] = el.access
    #     return {key: access_convert_str_to_dict(val) for key, val in result.items()}

    @validates("username")
    def username_validate(self, key, value):
        return value.lower()

    @validates("email")
    def email_validate(self, key, value):
        return value.lower()


class Role(Base):
    __tablename__ = "role"
    __table_args__ = {"comment": "Таблица видов ролей для авторизации"}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    name = Column(String(80), comment="Наименование Роли")

    role_item = relationship("RoleItem", cascade="all,delete", backref="role_item")


class RoleItem(Base):
    __tablename__ = "role_item"
    __table_args__ = (
        UniqueConstraint("role_id", "item", name="_role_item_uc"),
        {"comment": "Таблица объектов (таблиц) доступных Роли"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    role_id = Column(ForeignKey("role.id"), nullable=False, comment="ID Роли")
    item = Column(String(40), nullable=False, comment="Объект (таблица) доступный Роли")
    access = Column(String(20), nullable=False, comment="Строка доступа (CRUD+массовые операции)")

    # role = relationship("Role", viewonly=True, lazy="joined")

    @validates("item")
    def item_validate(self, key, value):
        if value.lower() not in get_tables_tracked_for_authorization():
            raise ValueError(f'Incorrect "item" (table name) = "{value}"!')
        return value.lower()

    @property
    def access_dict(self):
        return {
            el.value: (True if str(self.access + "0" * 20)[count] == "1" else False)
            for count, el in enumerate(TypeCRUD)
        }

    @access_dict.setter
    def access_dict(self, value):
        self.access = "".join(["1" if (el.value in value and value[el.value]) else "0" for el in TypeCRUD])


class UserRole(Base):
    __tablename__ = "user_role"
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="_user_role_uc"),
        {"comment": "Таблица связи видов доступных Пользователю Ролей"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    user_id = Column(ForeignKey("user.id"), nullable=False, comment="ID Пользователя")
    role_id = Column(ForeignKey("role.id"), nullable=False, comment="ID Роли")

    # user = relationship("User", viewonly=True, lazy="joined")
    role = relationship("Role", viewonly=True, lazy="joined")


class UserHistory(Base):
    __tablename__ = "user_history"
    __table_args__ = {"comment": "Логи истории активностей Пользователей"}

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True, comment="ID")
    date_time = Column(DateTime, nullable=False, default=datetime.datetime.now, comment="Дата/время")
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, comment="ID пользователя")
    description = Column(String, index=True)

    # user = relationship("User", cascade="all,delete", back_populates="user_history")


class UserLoginHistory(Base):
    __tablename__ = "user_login_history"
    __table_args__ = {"comment": "История попыток регистрации Пользователей"}

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True, comment="ID")
    date = Column(DateTime, nullable=False, default=datetime.datetime.now, comment="Дата/время")
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, comment="ID пользователя")
    is_success = Column(Boolean, nullable=False, comment="Успешная регистрация (да/нет)")

    # user = relationship("User", viewonly=True, lazy="joined")


class TypeCRUD(enum.Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOAD = "load"
    EXPORT = "export"


TYPE_CRUD_COMMENT = {
    TypeCRUD.CREATE: "Создание",
    TypeCRUD.READ: "Чтение",
    TypeCRUD.UPDATE: "Изменение",
    TypeCRUD.DELETE: "Удаление",
    TypeCRUD.LOAD: "Пакетная загрузка из файла",
    TypeCRUD.EXPORT: "Экспорт в файл",
}


def access_convert_str_to_dict(access: str) -> dict:
    return {el.value: (True if str(access + "0" * 20)[count] == "1" else False) for count, el in enumerate(TypeCRUD)}


def get_all_tables() -> list:
    tables = []
    for key, data in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if isinstance(data, DeclarativeMeta):
            if "metadata" in data.__dict__ and "tables" in data.metadata.__dict__:
                tables = list(data.metadata.tables)
                break
    return tables
