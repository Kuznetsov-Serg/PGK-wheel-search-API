import asyncio

import numpy as np
import random
import sys
import types
import warnings
from copy import deepcopy
from datetime import datetime, date

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.params import Depends, File
from pandas import DataFrame
from pydantic import BaseModel
from sqlalchemy import UniqueConstraint, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, DeclarativeMeta
from starlette.background import BackgroundTasks
from starlette.responses import StreamingResponse
from typing import Any, Optional, Union

from app.api.deps import get_db, get_engine
from app.auth.models import TypeCRUD
from app.auth.token_ldap import check_token, CheckAccess, write_user_history
from app.settings import PARSED_CONFIG, EXCEL_MEDIA_TYPE
from app.utils.log_db import LogDB, MyLogTypeEnum
from app.utils.utils import (
    table_writer,
    read_excel_with_find_headers,
    save_df_with_unique,
    is_excel_by_content_type,
    df_replace_true_false_with_yes_no,
    dict_from_db,
    list_of_dict_from_sql_row,
)
from app.utils.utils_schema import SchemaGenerate, SimpleScheme
from app.utils.utils_sql import BuildSQL


class OurBaseModel(BaseModel):
    class Config:
        from_attributes = True
        # orm_mode = True


class CreateBoundMethod:
    def __init__(self, f, instance, name=None):
        """
        return a function with same code, globals, defaults, closure, and
        name (or provide a new name)
        """
        fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)
        # in case f was given attrs (note this dict is a shallow copy):
        fn.__dict__.update(f.__dict__)
        fn.__annotations__ = deepcopy(f.__annotations__)
        fn.__kwdefaults__ = deepcopy(f.__kwdefaults__)
        self.__func__, self.__self__ = fn, instance

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


def get_obj_by_name(name_str):
    """
    getobj('scipy.stats.stats') returns the associated module
    getobj('scipy.stats.stats.chisquare') returns the associated function
    """
    try:
        return globals()[name_str]
    except KeyError:
        try:
            return __import__(name_str, fromlist=[""])
        except ImportError:
            modname, _, basename = name_str.rpartition(".")
            if modname:
                mod = get_obj_by_name(modname)
                return getattr(mod, basename)
            else:
                raise


class SchemaCRUD(OurBaseModel):
    """
    A schema that optionally contains useful refinements for configuring the behavior of EndPoints and Methods.
    :param sort_model - Example [{"col_id": "date", "sort": "desc"}, {"col_id": "name", "sort": "asc"}, {"col_id": "storage"}]
    :param filter_by - Example "name"
    :param order_by - If the value is None, get the value from 'filter_by'. If 'filter_by' = None too, it is equal to 'id'. Example "date_detect"
    """

    sort_model: Optional[list[dict]] = None
    filter_by: str = None
    order_by: str = None


class UniversalCRUD:
    """
    A universal class that supports the basic CRUD + Import + Export methods for working with any Model.
    The MetaClass dynamically creates instances of the class and EndPoints.
    At the same time, the methods are redefined in accordance with the annotations of the Model and Schemas.
    :param model - Model of DataBase
    :param router_lst - if the values is not None, the class creates an instance of the router with endpoints and adds the router to the List.
    :type router_lst - List of Router, optional, defaults to None.
    :param config - A dict that optionally contains useful refinements for configuring the behavior of EndPoints and Methods.
    :param schema - the schema includes all fields of the model with types, annotations, and required attributes.
    If the value is None, 'schema' will be created dynamically using the SchemaGenerate class.
    :param schema_create - identical to 'schema' but does not contain keys created automatically
    :param schema_update - identical to 'schema_create', but all fields are optional (the so-called PARTIAL_SCHEMA)
    """

    def __init__(
        self,
        model: DeclarativeMeta,
        router_lst: list[APIRouter] = None,
        config: SchemaCRUD = None,
        schema: BaseModel = None,
        schema_create: BaseModel = None,
        schema_update: BaseModel = None,
    ) -> None:
        self.model = model
        self.table_name = self.model.__tablename__
        self.path_starts_with = self.table_name.replace("_", "-")
        self.table_comment = self._get_table_comment()
        self.columns_unique = self._get_columns_unique()
        self.columns_required = self._get_columns_required()

        if router_lst is not None:
            self.router = APIRouter(
                prefix=PARSED_CONFIG.api_prefix,
                # dependencies=[Depends(check_token)],
                tags=[f'Endpoints for "{self.table_name}" {f"({self.table_comment})" if self.table_comment else ""}'],
            )
            router_lst.append(self.router)
        else:
            self.router = None

        self.schema_cls = SchemaGenerate(self.model, schema, schema_create, schema_update)
        self.schema = self.schema_cls.schema
        self.schema_create = self.schema_cls.schema_create
        self.schema_update = self.schema_cls.schema_update
        self.sort_model = config.get("sort_model") if config else None
        self.filter_by = config.get("filter_by") if config else None
        self.order_by = config.get("order_by", (self.filter_by or "id")) if config else "id"
        if self.router:
            self._create_endpoints()

    def _create_endpoints(self):
        # create dependencies
        self.check_access_create = CheckAccess(table_name=self.table_name, type_crud=TypeCRUD.CREATE)
        self.check_access_read = CheckAccess(table_name=self.table_name, type_crud=TypeCRUD.READ)
        self.check_access_update = CheckAccess(table_name=self.table_name, type_crud=TypeCRUD.UPDATE)
        self.check_access_delete = CheckAccess(table_name=self.table_name, type_crud=TypeCRUD.DELETE)
        self.check_access_load = CheckAccess(table_name=self.table_name, type_crud=TypeCRUD.LOAD)
        self.check_access_export = CheckAccess(table_name=self.table_name, type_crud=TypeCRUD.EXPORT)

        self.router.add_api_route(
            f"/{self.path_starts_with}-load/",
            name=f"Load from excel-file (all fields)",
            description=f'Loading the records in "{self.table_name}" from excel-file with all fields.',
            endpoint=self.endpoint_load,
            methods=["POST"],
            dependencies=[Depends(self.check_access_load)],
        )
        # creating a new instance of the bound method
        self.endpoint_create = types.MethodType(self._copy_func(self.endpoint_create), self)
        self.endpoint_create.__annotations__["row"] = self.schema_create  # set correct annotation
        self.router.add_api_route(
            f"/{self.path_starts_with}/",
            response_model=self.schema,
            name="Create record",
            description=f'Creating a record in "{self.table_name}".',
            endpoint=self.endpoint_create,
            methods=["POST"],
            dependencies=[Depends(self.check_access_create)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}/" + "{row_id}",
            response_model=self.schema,
            name=f"Get record",
            description=f'Get a record from "{self.table_name}" by "id".',
            endpoint=self.endpoint_get,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        # creating a new instance of the bound method
        self.endpoint_update = types.MethodType(self._copy_func(self.endpoint_update), self)
        self.endpoint_update.__annotations__["row"] = self.schema_update  # set correct annotation
        self.router.add_api_route(
            f"/{self.path_starts_with}/" + "{row_id}",
            response_model=self.schema,
            name=f"Update record",
            description=f'Update a record in "{self.table_name}".',
            endpoint=self.endpoint_update,
            methods=["PATCH"],
            dependencies=[Depends(self.check_access_update)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}/" + "{row_id}",
            name=f"Delete record",
            description=f'Delete a record in "{self.table_name}".',
            endpoint=self.endpoint_delete,
            methods=["DELETE"],
            dependencies=[Depends(self.check_access_delete)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-delete-advanced/",
            name=f"Delete list records with advanced parameters",
            description=f'Delete a list of records from "{self.table_name}" with advanced parameters.',
            endpoint=self.endpoint_delete_advanced,
            methods=["DELETE"],
            dependencies=[Depends(self.check_access_delete)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-list/",
            response_model=list[self.schema],
            name=f"Get list records",
            description=f'Get a list of records from "{self.table_name}".',
            endpoint=self.endpoint_list,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.endpoint_list_advanced = types.MethodType(self._copy_func(self.endpoint_list_advanced), self)
        self.router.add_api_route(
            f"/{self.path_starts_with}-list-advanced/",
            response_model=list[dict],
            name=f"Get list records with advanced parameters.",
            description=f'Get a list of records from "{self.table_name}" with advanced parameters.',
            endpoint=self.endpoint_list_advanced,
            methods=["PUT"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.endpoint_export = types.MethodType(self._copy_func(self.endpoint_export), self)
        self.router.add_api_route(
            f"/{self.path_starts_with}-export/",
            name=f"Export records to excel-file (all fields).",
            description=f'Exporting records (all fields) from "{self.table_name}" to excel-file.',
            endpoint=self.endpoint_export,
            methods=["PUT"],
            dependencies=[Depends(self.check_access_export)],
        )

    async def endpoint_load(
        self,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
        engine: Engine = Depends(get_engine),
        uploaded_file: UploadFile = File(...),
        is_overwrite: bool = False,
        is_async: bool = False,
        token=Depends(check_token),
    ) -> Any:
        username = token.get("sub", "NoAuthorised")
        if not is_excel_by_content_type(uploaded_file.content_type):
            raise HTTPException(
                status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
            )

        parent_name = sys._getframe().f_code.co_name  # function_name
        parent_id = -random.randint(1, 10000)  # send ID<0 for get new ID such as ID
        log_db = LogDB(db, username)
        log_db.put(
            parent_id=parent_id,
            parent_name=parent_name,
            type_log=MyLogTypeEnum.START,
            msg=f'Старт функции "{parent_name}" для обработки "{uploaded_file.filename}" ("{self.table_name}")',
        )
        args = (
            self.load,
            db,
            engine,
            uploaded_file,
            is_overwrite,
            username,
            log_db,
            is_async,
        )
        if is_async:
            background_tasks.add_task(*args)
            result = f'Функция "{parent_name}" для обработки "{uploaded_file.filename}" ({self.table_name}) запущена в фоновом режиме'
        else:
            loop = asyncio.get_running_loop()
            result = await asyncio.gather(loop.run_in_executor(None, *args))
        result = {"message": result[0] if type(result) is list else result, "log_id": log_db.id}
        write_user_history(
            db=db,
            username=username,
            message=f'Функцией "{parent_name}" обработан "{uploaded_file.filename}" ({result})',
        )
        return result

    async def endpoint_load_special(
        self,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
        engine: Engine = Depends(get_engine),
        uploaded_file: UploadFile = File(...),
        is_overwrite: bool = False,
        is_async: bool = False,
        token=Depends(check_token),
    ) -> Any:
        username = token.get("sub", "NoAuthorised")
        if not is_excel_by_content_type(uploaded_file.content_type):
            raise HTTPException(
                status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
            )
        parent_name = sys._getframe().f_code.co_name  # function_name
        parent_id = -random.randint(1, 10000)  # send ID<0 for get new ID such as ID
        log_db = LogDB(db, username)
        log_db.put(
            parent_id=parent_id,
            parent_name=parent_name,
            type_log=MyLogTypeEnum.START,
            msg=f'Старт функции "{parent_name}" для обработки "{uploaded_file.filename}" ("{self.table_name}")',
        )
        args = (
            self.load_special,
            db,
            engine,
            uploaded_file,
            is_overwrite,
            username,
            log_db,
            is_async,
        )
        if is_async:
            background_tasks.add_task(*args)
            result = f'Функция "{parent_name}" для обработки "{uploaded_file.filename}" ({self.table_name}) запущена в фоновом режиме'
        else:
            loop = asyncio.get_running_loop()
            result = await asyncio.gather(loop.run_in_executor(None, *args))
        result = {"message": result[0] if type(result) is list else result, "log_id": log_db.id}
        write_user_history(
            db=db,
            username=username,
            message=f'Функцией "{parent_name}" обработан "{uploaded_file.filename}" ({result})',
        )
        return result

    async def endpoint_create(self, row: SimpleScheme, db: Session = Depends(get_db), token=Depends(check_token)):
        result = await self.create_record(db, row)
        # write_user_history(
        #     db=db, username=username, message=f'Called "{sys._getframe().f_code.co_name}" for "{self.table_name}"'
        # )
        if PARSED_CONFIG.is_log_crud_change:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'В таблицe `{self.table_name}` создана запись:\n{dict_from_db(result)}',
            )
        return result

    async def endpoint_get(self, row_id: int, db: Session = Depends(get_db), token=Depends(check_token)):
        # , user_id: Optional[str] = Cookie(None)):
        result = await self.get_record(db, row_id)
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Просмотр записи из таблицы `{self.table_name}`:\n{dict_from_db(result)}',
            )
        return result

    async def endpoint_update(
        self,
        row_id: int,
        row: SimpleScheme,
        db: Session = Depends(get_db),
        token=Depends(check_token),
    ):
        if PARSED_CONFIG.is_log_crud_change:
            old_record = dict_from_db(await self.get_record(db, row_id))
        result = await self.update_record(db, row_id, row)
        if PARSED_CONFIG.is_log_crud_change:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Изменение записи в таблице `{self.table_name}`:\n'
                        f'было: {old_record}\nстало:{dict_from_db(result)}',
            )
        return result

    async def endpoint_delete(self, row_id: int, db: Session = Depends(get_db), token=Depends(check_token)):
        if PARSED_CONFIG.is_log_crud_change:
            old_record = dict_from_db(await self.get_record(db, row_id))
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Удаление записи в таблице `{self.table_name}`:\n{old_record}',
            )
        result = await self.delete_record(db, row_id)
        return result

    async def endpoint_delete_advanced(
        self,
        db: Session = Depends(get_db),
        id_list: Optional[list[int]] = None,
        filter_model: Optional[dict] = None,
        token=Depends(check_token),
    ):
        result = await self.delete_advanced(db, id_list, filter_model)
        if PARSED_CONFIG.is_log_crud_change:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Массовое удаление записей в таблице `{self.table_name}` с параметрами:'
                        f'\nid_list: {id_list}\nfilter_model: {filter_model}',
            )
        return result

    async def endpoint_list(
        self,
        filter_by: str = "",
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db),
        token=Depends(check_token),
    ):
        result = await self.get_list(db, filter_by, skip=skip, limit=limit)
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Просмотр списка записей в таблице `{self.table_name}` с параметрами:'
                        f'\nfilter_by: {filter_by}\nskip: {skip}\nlimit: {limit}',
            )
        return result

    async def endpoint_list_advanced(
        self,
        db: Session = Depends(get_db),
        filter_model: Optional[dict] = None,
        sort_model: Optional[list[dict]] = None,
        skip: int = 0,
        limit: int = 100,
        token=Depends(check_token),
    ):
        result = await self.get_list_advanced(db, filter_model, sort_model, skip, limit)
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Просмотр списка записей в таблице `{self.table_name}` с параметрами:'
                        f'\n- filter_model: {filter_model}\n- skip: {skip}\n- limit: {limit}',
            )
        return result

    async def endpoint_export(
        self,
        db: Session = Depends(get_db),
        filter_model: Optional[dict] = None,
        sort_model: Optional[list[dict]] = None,
        skip: int = 0,
        limit: int = 1000000,
        token=Depends(check_token),
    ):
        file_result = await self.export(db, filter_model=filter_model, sort_model=sort_model, skip=skip, limit=limit)
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Экспорт данных из таблицы `{self.table_name}` с параметрами:'
                        f'\n- filter_model: {filter_model}\n- skip: {skip}\n- limit: {limit}',
            )
        response = StreamingResponse(iter([file_result.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
        file_name = f"{self.table_name}_{date.today()}.xlsx"
        response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
        response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return response

    def load(
        self,
        db: Session,
        engine: Engine,
        uploaded_file: UploadFile,
        is_overwrite=True,
        username: str = "",
        log_db: LogDB = None,
        is_background: bool = False,
    ):
        columns_mapping = self._model_columns(False)
        columns_mapping_reverse = {value: key for key, value in columns_mapping.items()}

        time_start = datetime.now()
        content = uploaded_file.file.read()  # async read
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = read_excel_with_find_headers(
                    file=content,
                    headers_list=columns_mapping.values(),
                    headers_list_min=[columns_mapping[key] for key in self.columns_required],
                )
            except Exception as e:
                msg = e.detail if type(e) is HTTPException else f"Некорректный формат файла - {str(e)}"
                log_db.add(msg=msg, type_log=MyLogTypeEnum.ERROR)
                if is_background:
                    return
                raise HTTPException(status_code=422, detail=msg + f' (файл: "{uploaded_file.filename}")')

        report_df.columns = report_df.columns.str.replace("\n", " ").str.strip()
        log_db.add(msg=f"Из excel-файла считано {report_df.shape[0]} строк.")

        # removing unnecessary spaces
        report_df = report_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        report_df.rename(columns=columns_mapping_reverse, inplace=True)
        self._prepare_df_to_save(report_df)

        # Del old records if 'is_overwrite' = True
        amount_delete = 0
        if is_overwrite:
            for columns_unique_el in self.columns_unique:
                for i, el in report_df.iterrows():
                    filter_args = [self.model.__dict__[col] == el[col] for col in columns_unique_el]
                    amount_delete += db.query(self.model).filter(*filter_args).delete(synchronize_session="fetch")
                    db.commit()

        # Add new row
        columns_unique_el = self.columns_unique[0] if self.columns_unique else []
        amount_add = save_df_with_unique(
            db,
            engine,
            self.table_name,
            report_df,
            unique_cols=self.columns_unique,
            cols=list(columns_mapping.keys()),
        )
        amount_delete = f", удалено - {amount_delete} записей" if amount_delete else ""
        message = (
            f'В результате обработки "{uploaded_file.filename}" было {amount_add}'
            f'{amount_delete} (период выполнения - {str(datetime.now() - time_start).split(".", 2)[0]}).'
        )
        log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
        return message

    def load_special(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    async def get_list(self, db: Session, filter_by: str = "", skip: int = 0, limit: int = 100):
        if self.filter_by:
            if self.model.__dict__[self.filter_by].expression.type.python_type == int:
                if filter_by == "":
                    return (
                        db.query(self.model)
                        .order_by(self.model.__dict__[self.order_by].name)
                        .offset(skip)
                        .limit(limit)
                        .all()
                    )
                else:
                    return (
                        db.query(self.model)
                        .filter(self.model.__dict__[self.filter_by] == filter_by)
                        .order_by(self.model.__dict__[self.order_by].name)
                        .offset(skip)
                        .limit(limit)
                        .all()
                    )
            else:
                return (
                    db.query(self.model)
                    .filter(self.model.__dict__[self.filter_by].ilike(f"%{filter_by}%"))
                    .order_by(self.model.__dict__[self.order_by].name)
                    .offset(skip)
                    .limit(limit)
                    .all()
                )
        else:
            return (
                db.query(self.model).order_by(self.model.__dict__[self.order_by].name).offset(skip).limit(limit).all()
            )

    async def get_list_advanced(
        self,
        db: Session,
        filter_model: dict = dict(),
        sort_model: Union[dict, list] = dict(),
        skip: int = 0,
        limit: int = 100,
    ):
        # add filtering by any fields and sorting by any field and restrictions on the number starting from ...
        sql_command = (
            f'select * from "{self.table_name}" '
            f"{' where ' if filter_model and len(filter_model) else ''}"
            f"{BuildSQL.get_sql_condition(filter_model, sort_model, model=self.model)}"
            f"  LIMIT {limit} OFFSET {skip};"
        )
        try:
            result = db.execute(text(sql_command)).all()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error: {e}")
        return list_of_dict_from_sql_row(result)

    async def get_record(self, db: Session, row_id: int):
        result = db.query(self.model).filter(self.model.id == row_id).first()
        if result is None:
            raise HTTPException(status_code=404, detail=f'В "{self.table_name}" запись с id={row_id} не найдена')
        return result

    async def create_record(self, db: Session, record: Any):
        try:
            record_schema = self.schema_create(**record.dict())
        except Exception as err:
            raise HTTPException(status_code=409, detail=f"Error in validation of Scheme_create: {err}")
        try:
            db_result = self.model(**record.dict())
            db.add(db_result)
            db.commit()
            db.refresh(db_result)
        except Exception as err:
            raise HTTPException(status_code=409, detail=f"Error: {err}")
        return db_result

    async def update_record(self, db: Session, row_id: int, record: Any):
        try:
            record_schema = self.schema_update(**record.dict())
        except Exception as err:
            raise HTTPException(status_code=409, detail=f"Error in validation of Scheme_update: {err}")
        db_item = await self.get_record(db, row_id)
        try:
            # in this case, validation of fields in the model does not work with partial updates
            # query = db.query(self.model).filter(self.model.id == row_id).update(record.dict(exclude_unset=True))
            for key, val in record.dict(exclude_unset=True).items():
                setattr(db_item, key, val)  # in this case, validation works
            db.commit()
        except Exception as err:
            raise HTTPException(status_code=409, detail=f"Error: {err}")
        return await self.get_record(db, row_id)

    async def delete_record(self, db: Session, row_id: int):
        db_result = await self.get_record(db, row_id)
        try:
            db.delete(db_result)
            db.commit()
        except Exception as err:
            raise HTTPException(status_code=409, detail=f"Error: {err}")
        return {"message": "OK"}

    async def delete_advanced(self, db: Session, id_list: list[int] = None, filter_model: dict = dict()):
        # the "id_list" (list of IDs) is more important than "filter_model"
        # if id_list and id_list != [0] and len(id_list):
        if id_list and len(id_list):
            filter_model = {"id": {"type": "inList", "filter": id_list}}
        # add filtering by any fields
        sql_command = (
            f'delete from "{self.table_name}" '
            f"{' where ' if filter_model and len(filter_model) else ''}"
            f"{BuildSQL.get_sql_condition(filter_model, model=self.model)}"
        )
        try:
            result = db.execute(text(sql_command))
            db.commit()
        except Exception as err:
            raise HTTPException(status_code=409, detail=f"Error: {err}")
        return {"message": f"Deleted {result.rowcount} rows."}

    async def export(self, db: Session, *args, **kwargs):
        result = await self.get_list_advanced(db, *args, **kwargs)
        try:
            df = self._df_from_sql(result)
        except:
            df = DataFrame(result)

        # convert "True/False" to "да/нет"
        df = df_replace_true_false_with_yes_no(df, self.model)

        columns_sorted = [el.key for el in self.model.__table__.columns if el.key in df.columns]
        df = df[columns_sorted]
        df.rename(columns=self._model_columns(False), inplace=True)
        stream = table_writer(dataframes={f"{self.table_name}": df}, param="xlsx")
        return stream

    def _model_columns(self, is_with_id: bool = True, model_class: BaseModel = None) -> dict:
        model_class = model_class if model_class else self.model
        result = {
            el.key: (el.comment if el.comment else el.key)
            for el in model_class.__table__.columns
            if (is_with_id or el.key != "id")
        }
        # removing duplicates
        for k, v in result.items():
            count = list(result.values()).count(v)
            if count > 1:
                result[k] = f"{v}{'_' * (count - 1)}"
        return result

    def _copy_func(self, f, name: str = None):
        """
        return a function with same code, globals, defaults, closure, and
        name (or provide a new name)
        """
        __defaults__ = f.__defaults__
        if "sort_model" in f.__annotations__ and self.sort_model:
            __defaults__ = list(__defaults__)
            __defaults__[list(f.__annotations__).index("sort_model")] = self.sort_model
            __defaults__ = tuple(__defaults__)

        fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__, __defaults__, f.__closure__)
        # in case f was given attrs (note this dict is a shallow copy):
        fn.__dict__.update(f.__dict__)
        fn.__annotations__ = deepcopy(f.__annotations__)
        fn.__kwdefaults__ = deepcopy(f.__kwdefaults__)
        return fn

    def _prepare_df_to_save(self, df: DataFrame) -> Any:
        cols_db = {el.key: el.type for el in self.model.__table__.columns}
        for col in df.columns:
            try:
                # let's make the right types of columns
                if str(df[col].dtype) != "datetime64[ns]" and cols_db[col].python_type == date:
                    # df[col] = df[col].fillna('').map(lambda x: parse(str(x)).date() if x else np.nan)
                    df[col] = df[col].astype("datetime64[ns]")
                if cols_db[col].python_type == bool:
                    if str(df[col].dtype) == "object":
                        df.loc[df[col].str.upper() == "ДА", col] = True
                        df.loc[df[col] != True, col] = False
                        df[col] = df[col].astype(bool)
                    if str(df[col].dtype) != "bool":
                        df[col] = df[col].astype(bool)
                # if str(df[col].dtype) != "object" and cols_db[col].python_type == str:
                #     df[col] = df[col].astype(str)
                # cut the length if it is longer than in the database
                if str(df[col].dtype) in ("object", "string"):
                    max_len = df[col].str.len().max()
                    max_len = 0 if max_len is np.nan else int(max_len)
                    if (
                        cols_db[col].python_type == str
                        and str(cols_db[col]) != "TEXT"
                        and cols_db[col].length < max_len
                    ):
                        df[col] = df[col].apply(lambda x: x[: cols_db[col].length])
            except Exception as err:
                pass
        # remove duplicates according to the uniqueness rules
        for el in self.columns_unique:
            df.drop_duplicates(subset=el, inplace=True)
        return

    def _get_columns_unique(self) -> list:
        result = []
        if "__table_args__" in self.model.__dict__:
            for el in self.model.__table_args__:
                if isinstance(el, UniqueConstraint):
                    result.append([col.key for col in el.columns._all_columns])
        columns_unique = [el.key for el in self.model.__table__.columns if el.unique]
        result.extend([[el] for el in columns_unique])
        return result

    def _get_columns_required(self) -> list:
        result = []
        if "__table_args__" in self.model.__dict__:
            for el in self.model.__table_args__:
                if isinstance(el, UniqueConstraint):
                    result.extend([col.key for col in el.columns._all_columns])
        columns_required = [
            el.key for el in self.model.__table__.columns if not el.nullable or el.unique or el.foreign_keys
        ]
        columns_key = [el.key for el in self.model.__table__.columns if el.primary_key and el.autoincrement]
        result.extend([el for el in columns_required])
        result = list(set(result) - set(columns_key))
        return result

    def _get_table_comment(self) -> str:
        table_comment = ""
        if "__table_args__" in self.model.__dict__:
            if isinstance(self.model.__table_args__, dict):
                table_comment = self.model.__table_args__.get("comment")
            else:
                for el in self.model.__table_args__:
                    if isinstance(el, dict):
                        table_comment = el.get("comment")
        return table_comment

    @staticmethod
    def _df_from_sql(query):
        try:
            return (
                DataFrame(
                    [{key: val for key, val in zip(el._fields, el._data)} for el in query]) if query else DataFrame()
            )
        except:
            return DataFrame(query)
        # return DataFrame([el.__dict__ for el in query]).drop(columns="_sa_instance_state") if query else DataFrame()
