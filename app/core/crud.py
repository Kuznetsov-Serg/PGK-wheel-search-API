import random
import re
import sys
import warnings
from datetime import date, datetime

import numpy as np
from dateutil.parser import parse
from typing import Optional, Union, Any

import Levenshtein
import pandas as pd
import swifter # NB! don't delete
from fastapi import HTTPException, UploadFile, Query, Depends, File, Body
from pandas import DataFrame
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.util import asyncio
from starlette.background import BackgroundTasks
from starlette.responses import StreamingResponse
from tqdm import tqdm

from ..api.deps import get_db, get_engine, get_engine_ora
from ..auth.token_ldap import write_user_history, check_token
from ..settings import SourceType, PARSED_CONFIG, EXCEL_MEDIA_TYPE
from ..utils.log_db import LogDB, MyLogTypeEnum
from ..utils.utils import (
    read_excel_with_find_headers,
    save_df_with_unique,
    table_writer,
    is_excel_by_content_type,
    transliteration,
    df_replace_true_false_with_yes_no, df_column_excel_int_to_date,
)
from . import models
from ..utils.utils_crud import UniversalCRUD
from ..utils.utils_sql import BuildSQL


async def work_load(db: Session, engine: Engine, uploaded_file: UploadFile, is_remove_current: bool = False):
    sheet_work_filter = "Перечень наименований работ"
    sheet_work_cost = "Цены без работ"
    try:
        content = await uploaded_file.read()  # async read
        sheet_dict = pd.read_excel(content, None)  # Get the names of the sheets

        sheet_work_filter = sheet_work_filter if sheet_work_filter in list(sheet_dict) else list(sheet_dict)[0]
        sheet_work_cost = sheet_work_cost if sheet_work_cost in list(sheet_dict) else list(sheet_dict)[1]

        work_filter_df = sheet_dict.get(sheet_work_filter)
        work_cost_df = sheet_dict.get(sheet_work_cost)

        work_filter_df.rename(columns={"Наименование работы в договоре": "name"}, inplace=True)
        work_cost_df.rename(
            columns={"Наименование работы": "name", "Внешний номер договора": "number", "Цена": "cost"}, inplace=True
        )

        # reformat incorrect data in float column
        work_cost_df["cost"] = pd.to_numeric(
            work_cost_df["cost"].apply(
                lambda x: x.replace(",", ".").replace("\xa0", "").replace(" ", "") if isinstance(x, str) else x
            )
        )

        if is_remove_current:
            db.execute(f"TRUNCATE work_filter, work_cost RESTART IDENTITY CASCADE;")
            db.commit()

        # Add new row in filter_work
        work_filter_add = save_df_with_unique(db, engine, "work_filter", work_filter_df, unique_cols=["name"])
        # Add new row in add_work
        work_cost_add = save_df_with_unique(
            db, engine, "work_cost", work_cost_df, unique_cols=["name", "number", "cost"]
        )

        result = {
            "msg": f'Успешная обработка справочников "filter_work" ({work_filter_add}), '
            f'и "add_work" ({work_cost_add}).'
        }
        # result = {"msg": f"In filter_work ({work_filter_add}), in add_work ({work_cost_add})."}
    except:
        raise HTTPException(status_code=422, detail=f"Некорректный формат файла")
        # raise HTTPException(status_code=422, detail=f"Incorrect file format")

    return result


class StealCRUD(UniversalCRUD):
    class UniqueId(BaseModel):
        document_num: int
        document_num_num: int

    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-list-special/",
            response_model=list[self.schema],
            name=f"Get list records (outdated)",
            description=f'Get a list of records from "{self.table_name}".',
            endpoint=self.endpoint_list_special,
            methods=["POST"],
            dependencies=[Depends(self.check_access_read)],
        )

        self.router.add_api_route(
            f"/{self.path_starts_with}-load-special/",
            name=f"Load from excel-file (as the user requested)",
            description=f'Loading the records in "{self.table_name}" from excel-file (as the user requested).',
            endpoint=self.endpoint_load_special,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/check-spr-on-steal/",
            name="Checking WorkFilter and WorkCost for compliance with Steal (excel-files)",
            description="The `WorkFilter` and `WorkCost` analysis function compares rows from 'Steal' (Excel) using the Levenshtein algorithm.",
            endpoint=self.endpoint_check_spr_on_steal,
            methods=["POST"],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-export-special/",
            name=f"Export records to excel-file (as the user requested).",
            description=f'Exporting records from "{self.table_name}" to excel-file (as the user requested).',
            endpoint=self.endpoint_export_special,
            methods=["PUT"],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-generate-unique-id/",
            response_model=self.UniqueId,
            name=f"Generate unique ID",
            description=f'Generate unique "id" for "{self.table_name}".',
            endpoint=self.endpoint_generate_unique_id,
            methods=["GET"],
        )

    async def endpoint_list_special(
            self,
            date_from: Optional[date] = Query(None, title="date from",
                                              description="'Date from' (it is used if not empty) - `YYYY-MM-DD`"),
            date_to: Optional[date] = Query(None, title="date to",
                                            description="'Date to' (it is used if not empty) - `YYYY-MM-DD`"),
            is_include_date_null: Optional[bool] = Query(False,
                                                         description="True - include rows with a 'date_detect' = Null"),
            filter_model: Optional[dict] = None,
            sort_model=Body({"col_id": "date_detect", "sort": "desc"}),
            skip: int = 0,
            limit: int = 100,
            db: Session = Depends(get_db),
            token=Depends(check_token),
    ):
        result = await self.get_list_special(
            db, date_from, date_to, is_include_date_null, filter_model, sort_model, skip, limit
        )
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Просмотр списка записей в таблице `{self.table_name}` с параметрами:'
                        f'\ndate_from: {date_from}\ndate_to: {date_to}\nskip: {skip}\nlimit: {limit}',
            )
        return result

    async def endpoint_load_special(
        self,
        background_tasks: BackgroundTasks,
        files_list: list[UploadFile],
        # files_list: UploadFile = File(media_type='text/xml'),
        period_load: Optional[str] = Query(
            f"{date.today().year}-{date.today().month}",
            title="period load",
            description='"Период загрузки" в формате "YYYY-MM" (если "Дата выявления разоборудования (ремонта вагона)" '
            "пустая, берется Вами введенное значение)",
        ),
        is_overwrite: bool = False,
        is_async: bool = False,
        db: Session = Depends(get_db),
        engine: Engine = Depends(get_engine),
        token = Depends(check_token),
    ) -> Any:
        # checking for valid file types
        for uploaded_file in tqdm(files_list):
            if not is_excel_by_content_type(uploaded_file.content_type):
                raise HTTPException(
                    status_code=422,
                    detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")',
                )
        loop = asyncio.get_running_loop()
        username = token.get("sub", "NoAuthorised")
        result_all = []
        parent_name = sys._getframe().f_code.co_name  # function_name
        for uploaded_file in tqdm(files_list):
            parent_id = -random.randint(1, 10000)  # send ID<0 for get new ID such as ID
            log_db = LogDB(db)
            log_db.put(
                parent_id=parent_id,
                parent_name=parent_name,
                type_log=MyLogTypeEnum.START,
                msg=f'Старт функции "{parent_name}" для обработки "{uploaded_file.filename}"',
                # is_append=False,
                username=username,
            )
            args = self.load_special, db, engine, uploaded_file, period_load, is_overwrite, username, log_db, is_async
            if is_async:
                background_tasks.add_task(*args)
                result = f'Функция "steal_load" для обработки "{uploaded_file.filename}" запущена в фоновом режиме'
            else:
                result = await asyncio.gather(loop.run_in_executor(None, *args))
            result = {"message": result[0] if type(result) is list else result, "log_id": log_db.id}
            result_all.append(result)
            write_user_history(
                db=db,
                username=username,
                message=f'Функцией "{sys._getframe().f_code.co_name}" обработан "{uploaded_file.filename}" ({result})',
            )
        return result_all

    async def endpoint_check_spr_on_steal(
            self,
            files_list: list[UploadFile],
            is_overwrite: bool = True,
            amount_difference: int = 3,
            amount_repeat: int = 3,
            db: Session = Depends(get_db),
            engine: Engine = Depends(get_engine),
    ) -> Any:
        loop = asyncio.get_running_loop()
        username = PARSED_CONFIG.username

        result_all = []
        if is_overwrite:
            amount_delete = db.query(models.CheckFilter).delete(synchronize_session="fetch")
            db.commit()
            result_all.append(f"Delete {amount_delete} records.")

        for report in tqdm(files_list):
            uploaded_file = report.file
            result = await asyncio.gather(
                loop.run_in_executor(
                    None, self.check_spr_on_steal, engine, uploaded_file, amount_difference, amount_repeat
                ),
            )
            result_all.append(result)
            write_user_history(
                db=db,
                username=username,
                message=f'Функцией "check_spr_on_steal" обработан "{report.filename}" ({result})',
                # message=f'Called "check_spr_on_steal" from file="{report.filename}" ({result})',
            )

        return result_all

    @staticmethod
    async def get_list_special(
            db: Session,
            date_from: Optional[date] = None,
            date_to: Optional[date] = None,
            is_include_date_null: Optional[bool] = False,
            filter_model: dict = dict(),
            sort_model: Union[dict, list] = dict(),
            skip: int = 0,
            limit: int = 100,
    ):
        is_take_null = " or date_detect is null" if is_include_date_null else ""
        date_from = date_from if date_from else date(1900, 1, 1)
        date_to = date_to if date_to else date(2100, 1, 1)
        sql_command = f"select * from steal " f"where (date_detect between '{date_from}' and '{date_to}'{is_take_null})"
        # add filtering by any fields and sorting by any field and restrictions on the number starting from ...
        sql_command += (
            f"{' and ' if filter_model and len(filter_model) else ''}"
            f"{BuildSQL.get_sql_condition(filter_model, sort_model, 'date_detect')}"
            f"  LIMIT {limit} OFFSET {skip};"
        )
        try:
            result = db.execute(sql_command).all()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error: {e}")
        return result

    @staticmethod
    def load_special(
            db: Session,
            engine: Engine,
            uploaded_file: UploadFile,
            period_load: str,
            is_overwrite=True,
            username: str = "",
            log_db: LogDB = None,
            is_background: bool = False,
    ):
        def get_period_load(row):
            return period_load if pd.isnull(row.date_detect) else f"{row.date_detect.year}-{row.date_detect.month:02}"

        def is_filter_by_name(row, name_df):
            for i, el in name_df.iterrows():
                # if re.search(el["name"].replace("(", "\(").replace(")", "\)"), row.part_name, re.IGNORECASE):
                if re.search(re.escape(el["name"]), row.part_name, re.IGNORECASE):
                    return True
            return False

        def is_filter_by_storage_and_number_rdv(row, storage_df: DataFrame, all_row_df: DataFrame):
            def is_another_work_by_document_enable(document_num: int, work_name: str):
                document_df = all_row_df.loc[all_row_df["document_num"] == document_num]
                for _index, _row in document_df.iterrows():
                    if re.search(re.escape(work_name), _row["part_name"], re.IGNORECASE):
                        return True
                return False

            is_work_enable = False
            for i, el in storage_df.iterrows():
                if re.search(re.escape(el["name"]), row.part_name, re.IGNORECASE):
                    if (
                            (
                                    str(el.branch_name or "") == ""
                                    or re.search(re.escape(el.branch_name), row.branch_name, re.IGNORECASE)
                            )
                            and (
                            str(el.storage_name or "") == ""
                            or re.search(re.escape(el.storage_name), row.storage_name, re.IGNORECASE)
                    )
                            and (
                            str(el.number_rdv or "") == ""
                            or re.search(el.number_rdv, str(row[COLUMN_NUMBER_RDV]), re.IGNORECASE)
                    )
                            and (
                            str(el.work_name_by_document or "") == ""
                            or is_another_work_by_document_enable(row.document_num, el.work_name_by_document)
                    )
                    ):
                        return True
                    is_work_enable = True
            return not is_work_enable

        COLUMN_NUMBER_RDV = "Номер позиции РДВ"
        USECOLS = [
            "Документ",
            "Номер услуги",
            "Версия документа",
            "Дата проводки",
            "Тип документа",
            "Наименование работы",
            "Цена",
            "Количество",
            "Номер вагона",
            "Внешний номер договора",
            "Наименование филиала",
            "Обозначение склада",
            COLUMN_NUMBER_RDV,
        ]
        FACT_MAPPING_NAME = {
            "Дата загрузки": "date_upload",
            "Документ": "document_num",
            "Номер услуги": "document_num_num",
            # "Версия документа": "document_version",
            "period_load": "period_load",
            "Дата проводки": "date_detect",
            # "Тип документа",
            "Наименование работы": "part_name",
            "Цена": "part_cost",
            "Количество": "part_amount",
            "Номер вагона": "wagon_num",
            "Внешний номер договора": "external_contract_num",
            "Наименование филиала": "branch_name",
            "Обозначение склада": "storage_name",
        }
        time_start = datetime.now()
        content = uploaded_file.file.read()  # async read
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = read_excel_with_find_headers(file=content, sheet_name="Ремонты", headers_list=USECOLS)
                # report_df = pd.read_excel(content, engine="openpyxl", sheet_name="Ремонты", usecols=USECOLS)
            except Exception as e:
                try:
                    report_df = read_excel_with_find_headers(file=content, headers_list=USECOLS)
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

        # filter by the conditions in the technical specification
        report_df = report_df.loc[(report_df["Тип документа"] == "РДВ")]
        # report_df = report_df.loc[(report_df["Тип документа"] == "РДВ") & (report_df["Дата проводки"].notnull())]
        log_db.add(msg=f'После фильтрации, по "РДВ", осталось {report_df.shape[0]} строк.')

        report_df.rename(columns=FACT_MAPPING_NAME, inplace=True)

        report_df = report_df[
            report_df.groupby(["document_num", "document_num_num"])["Версия документа"].transform(max)
            == report_df["Версия документа"]
            ]
        log_db.add(msg=f"После очистки версионности документов (берем последнюю), осталось {report_df.shape[0]} строк.")

        # Add empty 'part_name' from 'work_cost' by 'number' and 'cost'
        work_cost_df = pd.read_sql("select * from work_cost;", con=engine)
        report_df = report_df.merge(
            work_cost_df, how="left", left_on=["external_contract_num", "part_cost"], right_on=["number", "cost"]
        )
        report_df.loc[report_df["part_name"].isnull(), "part_name"] = report_df.loc[
            report_df["part_name"].isnull(), "name"]
        report_df["part_name"] = report_df["part_name"].astype(str)

        # Filter by 'work_filter' and 'work_cost'
        work_filter_df = pd.read_sql("SELECT name FROM work_filter UNION DISTINCT SELECT name FROM work_cost;",
                                     con=engine)
        # report_filtered_df = report_df.merge(work_filter_df, how="inner", left_on="part_name", right_on="name")
        report_df["is_filter_by_name"] = report_df.swifter.progress_bar(False).apply(
            lambda row: is_filter_by_name(row, work_filter_df), axis=1
        )
        report_filtered_df = report_df.loc[report_df["is_filter_by_name"]].copy()
        # report_filtered_df = report_df.merge(work_filter_df, how="inner", left_on="part_name", right_on="name")

        if report_filtered_df.empty:
            message = "После фильтрации, по наименованию деталей, осталось 0 строк."
            log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
            return message

        # Filter by 'work_filter' with Storage, Branch, Number_RDV and ... if necessary
        work_filter_storage_df = pd.read_sql(
            "select * from work_filter as wf "
            "inner join work_filter_by_storage as wfs "
            "on (wf.id = wfs.work_filter_id);",
            con=engine,
        )
        report_filtered_df["is_filter_by_storage"] = report_filtered_df.swifter.progress_bar(False).apply(
            lambda row: is_filter_by_storage_and_number_rdv(row, work_filter_storage_df, report_df), axis=1
        )
        report_filtered_df = report_filtered_df.loc[report_filtered_df["is_filter_by_storage"]].copy()
        if report_filtered_df.empty:
            message = "После фильтрации, по Складам, Номеру РДВ и т.д., осталось 0 строк."
            log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
            return message

        # add 'date_detect' from current or previous "Дата проводки"
        # report_filtered_df["date_detect"] = report_filtered_df.apply(lambda row: get_date_detect(row, report_df), axis=1)

        # convert Excel DATA (int) to a regular DATE if necessary
        report_filtered_df["date_detect"] = df_column_excel_int_to_date(report_filtered_df["date_detect"])

        # add 'period_load' from "Дата проводки" or incoming value 'period_load'
        report_filtered_df["period_load"] = report_filtered_df.swifter.apply(lambda row: get_period_load(row), axis=1)
        report_filtered_df["date_upload"] = date.today()
        log_db.add(msg=f"После фильтрации, по условиям из справочников, осталось {report_filtered_df.shape[0]} строк.")

        # Del old records if 'is_overwrite' = True
        amount_delete = 0
        if is_overwrite:
            step = 50
            for count in range(0, report_filtered_df.shape[0], step):
                amount_delete += (
                    db.query(models.Steal)
                    .filter(models.Steal.document_num.in_(report_filtered_df["document_num"][count: count + step]))
                    .delete(synchronize_session="fetch")
                )
                db.commit()

        # Add new row in Steal
        amount_add = save_df_with_unique(
            db,
            engine,
            "steal",
            report_filtered_df,
            unique_cols=["document_num", "document_num_num"],
            cols=list(FACT_MAPPING_NAME.values()),
        )
        amount_delete = f", удалено - {amount_delete} записей" if amount_delete else ""
        message = (
            f'В результате обработки "{uploaded_file.filename}" было добавлено {amount_add}'
            f'{amount_delete} (период выполнения - {str(datetime.now() - time_start).split(".", 2)[0]}).'
        )
        log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
        return message

    @staticmethod
    def check_spr_on_steal(engine: Engine, uploaded_file: UploadFile, amount_difference, amount_repeat):
        """
        The 'WorkFilter' and 'WorkCost' analysis function compares rows from `Steal` (Excel) using the Levenshtein algorithm.
        """
        time_start = datetime.now()
        content = uploaded_file.read()  # async read

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = pd.read_excel(
                    content, engine="openpyxl", sheet_name="Ремонты", usecols=["Наименование работы", "Тип документа"]
                )
            except ValueError as e:
                try:
                    report_df = pd.read_excel(content, engine="openpyxl",
                                              usecols=["Наименование работы", "Тип документа"])
                except ValueError as e:
                    raise HTTPException(status_code=422, detail=f"Incorrect file format \n(Error: {e})")

        print("Reading from excel-file: ", report_df.shape)
        print(report_df.head(5))

        # removing unnecessary spaces
        report_df = report_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # filter by the conditions in the technical specification
        report_df = report_df.loc[(report_df["Тип документа"] == "РДВ") & report_df["Наименование работы"].notnull()]

        print("After filter by the conditions in the technical specification: ", report_df.shape)
        print(report_df.head(5))

        report_df.rename(columns={"Наименование работы": "part_name"}, inplace=True)

        # Filter by 'work_filter'
        work_filter_df = pd.read_sql("SELECT name FROM work_filter;", con=engine)
        report_df = report_df.merge(work_filter_df, how="left", left_on="part_name", right_on="name")
        report_df = report_df.loc[report_df["name"].isnull()]
        report_df = report_df.groupby(["part_name"], as_index=False).size()

        result_df = pd.DataFrame()
        for row in tqdm(report_df.itertuples(index=False)):
            # result = process.extractOne(el, work_filter_df['name'])
            work_filter_df["distance"] = work_filter_df["name"].apply(Levenshtein.distance, s2=row.part_name)
            result_tmp = work_filter_df.loc[work_filter_df["distance"] <= amount_difference]
            if not result_tmp.empty:
                result_tmp["part_name"] = row.part_name
                result_tmp["size"] = row.size
                # result_tmp['distance'] = work_filter_df['name'].apply(Levenshtein.distance, s2=el)
                result_df = result_df.append(result_tmp, ignore_index=True)
        result_df = result_df.groupby(["name", "part_name", "distance"], as_index=False).agg({"size": "sum"})
        # result_df = result_df.groupby(['name', 'part_name', 'distance'], as_index=False).size()
        result_df = result_df.loc[result_df["size"] >= amount_repeat]
        result_df.rename(columns={"name": "filter_name", "part_name": "work_name", "size": "count"}, inplace=True)

        add_record = result_df.to_sql("check_filter", con=engine, if_exists="append", method="multi", index=False)

        return f'Add {add_record} records. (execution period {str(datetime.now() - time_start).split(".", 2)[0]})'

    async def endpoint_generate_unique_id(self, db: Session = Depends(get_db)):
        result = await self.generate_unique_id(db)
        return result

    async def generate_unique_id(self, db: Session):
        document_num = 0  # we will generate on the zero number
        document_num_num_max = 200000000  # limit int: 2 147 483 647
        row = db.execute(f"select max(document_num_num) from steal where document_num = {document_num};").all()
        document_num_num = row[0].max + 1 if row[0].max else 1
        if document_num_num > document_num_num_max:
            step = 2
            document_num_num = 0
            for skip in range(0, document_num_num_max, step):
                rows = db.execute(
                    f"select document_num_num from steal where document_num = {document_num} "
                    f"ORDER by document_num_num offset {skip} limit {step};"
                ).all()
                for el in rows:
                    document_num_num += 1
                    # print('skip=', skip, ' num=', document_num_num, ' num_bd=', el.document_num_num)
                    if document_num_num < el.document_num_num:
                        return {"document_num": document_num, "document_num_num": document_num_num}

            raise HTTPException(
                status_code=501,
                detail=f"Error: Ошибка генерации ИД " f"(все значения до {document_num_num_max} заняты)",
            )

        return {"document_num": document_num, "document_num_num": document_num_num}

    async def endpoint_export_special(
            self,
            date_from: Optional[date] = Query(None, title="date from",
                                              description="'Date from' (it is used if not empty) - `YYYY-MM-DD`"),
            date_to: Optional[date] = Query(None, title="date to",
                                            description="'Date to' (it is used if not empty) - `YYYY-MM-DD`"),
            is_include_date_null: Optional[bool] = Query(False,
                                                         description="True - include rows with a 'date_detect' = Null"),
            filter_model: Optional[dict] = None,
            sort_model=Body({"col_id": "date_detect", "sort": "desc"}),
            skip: int = 0,
            limit: int = 1000000,
            db: Session = Depends(get_db),
            token=Depends(check_token),
    ):
        file_result = await self.export_special(db, date_from, date_to, is_include_date_null, filter_model, sort_model,
                                                skip, limit)
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Экспорт данных из таблицы `{self.table_name}` с параметрами:'
                        f'\n- filter_model: {filter_model}\n- skip: {skip}\n- limit: {limit}',
            )
        response = StreamingResponse(iter([file_result.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
        d1 = date_from.strftime("%Y_%m_%d") if date_from else ""
        d2 = date_to.strftime("%Y_%m_%d") if date_to else ""
        file_name = f"steal_{d1}-{d2}.xlsx"
        response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
        response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return response

    async def export_special(self, *args, **kwargs):
        STEAL_MAPPING = {
            "period_load": "Период загрузки из SAP",
            "date_detect": "Дата выявления разоборудования (ремонта вагона)",
            "branch_name": "Филиал",
            "storage_name": "Склад",
            "wagon_num": "Номер вагона",
            "part_name": "Наименование похищенной детали (номенклатура)",
            "part_cost": "Цена похищенных деталей, руб.",
            "part_amount": "Количество",
            "part_sum": "Стоимость похищенных деталей, руб.",
            "repay": "Всего возмещено, руб.",
            "repay_author": "Восстановление вагона за счёт ВРП/виновником, руб.",
            "insurance_notification_date": "Дата направления уведомления в СК",
            "insurance_notification_num": "Исходящий номер уведомления в СК",
            "insurance_name": "Наименование СК",
            "is_insurance_of_carrier": "Страховая компания перевозчика (да/нет)",
            "insurance_number": "Номер убытка (присваивается СК)",
            "insurance_claim_number": "Номер претензии",
            "insurance_payment_total": "Общий размер требований о выплате СК",
            "insurance_payment_date": "Дата выплаты страхового возмещения",
            "insurance_payment_done": "Выплаченная сумма СК, руб.",
            "author_pretension_date": "Дата направления претензии виновному",
            "author_name": "Наименование виновника",
            "author_pretension_number": "Исходящий номер претензии",
            "author_payment_total": "Общий размер требований о выплате",
            "author_payment_date": "Дата выплаты виновником по претензии",
            "author_payment_done": "Выплаченная сумма, руб.",
            "author_lawyer_date": "Дата передачи материала Юристам",
            "author_lawyer_number": "Номер Арбитражного дела",
            "police_date": "Дата передачи материала БР",
            "police_ovd_date": "Дата направления заявления в ОВД",
            "police_ovd_name": "Наименование ОВД",
            "police_payment": "Размер ущерба, руб.",
            "police_decision": "Процессуальное решение по заявлению (возбуждение уг. дела / отказ в ВУД)",
            "police_decision_date": "Дата процессуального решения",
        }

        result = await self.get_list_special(*args, **kwargs)
        report_df = self._df_from_sql(result)
        report_df["part_sum"] = report_df["part_cost"] * report_df["part_amount"]
        report_df.rename(columns=STEAL_MAPPING, inplace=True)
        report_df = report_df[STEAL_MAPPING.values()].sort_values(
            ["Период загрузки из SAP", "Дата выявления разоборудования (ремонта вагона)", "Филиал"]
        )

        stream = table_writer(dataframes={"Разоборудование вагонов": report_df}, param="xlsx")
        return stream  # .read()


class WorkFilterCRUD(UniversalCRUD):

    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            "/load-work-spr/",
            name="Downloading the directory of analyzed types of work and details with prices (Excel-file).",
            description="Downloading the directory of analyzed types of work and details with prices (Excel-file).",
            endpoint=self.endpoint_load_work_spr_and_work_cost,
            methods=["POST"],
        )

    async def endpoint_load_work_spr_and_work_cost(
            self,
            db: Session = Depends(get_db),
            engine: Engine = Depends(get_engine),
            uploaded_file: UploadFile = File(...),
            is_remove_current: bool = True,
            token=Depends(check_token),
    ) -> Any:
        # checking for valid file types
        if not is_excel_by_content_type(uploaded_file.content_type):
            raise HTTPException(
                status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
            )
        username = token.get("sub", "NoAuthorised")
        parent_name = sys._getframe().f_code.co_name  # function_name
        parent_id = -random.randint(1, 10000)  # send ID<0 for get new ID such as ID
        log_db = LogDB(db, username)
        log_db.put(
            parent_id=parent_id,
            parent_name=parent_name,
            type_log=MyLogTypeEnum.START,
            msg=f'Старт функции "{parent_name}" для обработки "{uploaded_file.filename}"',
            username=username,
        )

        result = await self.load_work_spr_and_work_cost(db, engine, uploaded_file, is_remove_current, log_db)
        # logger.info(f'User {PARSED_CONFIG.username} launched "load-filter-work-spr" (result={result})')
        write_user_history(
            db=db,
            username=username,
            message=f'Called "load_work_spr" from file="{uploaded_file.filename}" ({result})'
        )
        return result

    @staticmethod
    async def load_work_spr_and_work_cost(
            db: Session,
            engine: Engine,
            uploaded_file: UploadFile,
            is_remove_current: bool = False,
            log_db: LogDB = None,
    ):
        sheet_work_filter = "Перечень наименований работ"
        sheet_work_cost = "Цены без работ"
        try:
            content = await uploaded_file.read()  # async read
            sheet_dict = pd.read_excel(content, None)  # Get the names of the sheets

            sheet_work_filter = sheet_work_filter if sheet_work_filter in list(sheet_dict) else list(sheet_dict)[0]
            sheet_work_cost = sheet_work_cost if sheet_work_cost in list(sheet_dict) else list(sheet_dict)[1]

            work_filter_df = sheet_dict.get(sheet_work_filter)
            work_cost_df = sheet_dict.get(sheet_work_cost)

            work_filter_df.rename(columns={"Наименование работы в договоре": "name"}, inplace=True)
            work_cost_df.rename(
                columns={"Наименование работы": "name", "Внешний номер договора": "number", "Цена": "cost"}, inplace=True
            )

            # reformat incorrect data in float column
            work_cost_df["cost"] = pd.to_numeric(
                work_cost_df["cost"].apply(
                    lambda x: x.replace(",", ".").replace("\xa0", "").replace(" ", "") if isinstance(x, str) else x
                )
            )

            if is_remove_current:
                db.execute(f"TRUNCATE work_filter, work_cost RESTART IDENTITY CASCADE;")
                db.commit()

            # Add new row in work_filter
            work_filter_add = save_df_with_unique(db, engine, "work_filter", work_filter_df, unique_cols=["name"])
            # Add new row in work_cost
            work_cost_add = save_df_with_unique(
                db, engine, "work_cost", work_cost_df, unique_cols=["name", "number", "cost"]
            )
            message = f'Успешная обработка справочников `work_filter` ({work_filter_add}), ' \
                      f'и `work_cost` ({work_cost_add}).'
            log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)

        except Exception as err:
            message = err.detail if type(err) is HTTPException else f"Некорректный формат файла - {str(err)}"
            log_db.add(msg=message, type_log=MyLogTypeEnum.ERROR)
            raise HTTPException(status_code=422, detail=message)

        return message


class CheckFilterCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-check-filter/",
            # response_model=self.UniqueId,
            name=f"Checking WorkFilter for compliance with Steal (excel-files)",
            description=f'Checking WorkFilter for compliance with Steal (excel-files) "{self.table_name}".',
            endpoint=self.endpoint_check_filter_on_steal,
            methods=["POST"],
        )

    async def endpoint_check_filter_on_steal(
        self,
        files_list: list[UploadFile],
        is_overwrite: bool = True,
        amount_difference: int = 3,
        amount_repeat: int = 3,
        db: Session = Depends(get_db),
        engine: Engine = Depends(get_engine),
    ) -> Any:
        loop = asyncio.get_running_loop()
        username = PARSED_CONFIG.username

        result_all = []
        if is_overwrite:
            amount_delete = db.query(models.CheckFilter).delete(synchronize_session="fetch")
            db.commit()
            result_all.append(f"Delete {amount_delete} records.")

        for report in tqdm(files_list):
            uploaded_file = report.file
            result = await asyncio.gather(
                loop.run_in_executor(
                    None, self.check_filter_on_steal, db, engine, uploaded_file, amount_difference, amount_repeat
                ),
            )
            result_all.append(result)
            write_user_history(
                db=db,
                username=username,
                message=f'Функцией "{sys._getframe().f_code.co_name}" обработан "{report.filename}" ({result})',
                # message=f'Called "check_spr_on_steal" from file="{report.filename}" ({result})',
            )

        return result_all

    @staticmethod
    def check_filter_on_steal(db: Session, engine: Engine, uploaded_file: UploadFile, amount_difference, amount_repeat):
        """
        The 'WorkFilter' and 'WorkCost' analysis function compares rows from 'Steal' (Excel) using the Levenshtein algorithm.
        """
        time_start = datetime.now()
        content = uploaded_file.read()  # async read

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = pd.read_excel(
                    content, engine="openpyxl", sheet_name="Ремонты", LOADCOLS=["Наименование работы", "Тип документа"]
                )
            except ValueError as e:
                try:
                    report_df = pd.read_excel(
                        content, engine="openpyxl", LOADCOLS=["Наименование работы", "Тип документа"]
                    )
                except ValueError as e:
                    raise HTTPException(status_code=422, detail=f"Incorrect file format \n(Error: {e})")

        print("Reading from excel-file: ", report_df.shape)
        print(report_df.head(5))

        # removing unnecessary spaces
        report_df = report_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # filter by the conditions in the technical specification
        report_df = report_df.loc[(report_df["Тип документа"] == "РДВ") & report_df["Наименование работы"].notnull()]

        print("After filter by the conditions in the technical specification: ", report_df.shape)
        print(report_df.head(5))

        report_df.rename(columns={"Наименование работы": "part_name"}, inplace=True)

        # Filter by 'work_filter'
        work_filter_df = pd.read_sql("SELECT name FROM work_filter;", con=engine)
        report_df = report_df.merge(work_filter_df, how="left", left_on="part_name", right_on="name")
        report_df = report_df.loc[report_df["name"].isnull()]
        report_df = report_df.groupby(["part_name"], as_index=False).size()

        result_df = pd.DataFrame()
        for row in tqdm(report_df.itertuples(index=False)):
            # result = process.extractOne(el, work_filter_df['name'])
            work_filter_df["distance"] = work_filter_df["name"].apply(Levenshtein.distance, s2=row.part_name)
            result_tmp = work_filter_df.loc[work_filter_df["distance"] <= amount_difference]
            if not result_tmp.empty:
                result_tmp["part_name"] = row.part_name
                result_tmp["size"] = row.size
                # result_tmp['distance'] = work_filter_df['name'].apply(Levenshtein.distance, s2=el)
                result_df = result_df.append(result_tmp, ignore_index=True)
        result_df = result_df.groupby(["name", "part_name", "distance"], as_index=False).agg({"size": "sum"})
        # result_df = result_df.groupby(['name', 'part_name', 'distance'], as_index=False).size()
        result_df = result_df.loc[result_df["size"] >= amount_repeat]
        result_df.rename(columns={"name": "filter_name", "part_name": "work_name", "size": "count"}, inplace=True)

        add_record = result_df.to_sql("check_filter", con=engine, if_exists="append", method="multi", index=False)

        return f'Add {add_record} records. (execution period {str(datetime.now() - time_start).split(".", 2)[0]})'


class WheelSetCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-load-special/",
            name=f"Load from excel-file (as the user requested)",
            description=f'Loading the records in "{self.table_name}" from excel-file (as the user requested).',
            endpoint=self.endpoint_load_special,
            methods=["POST"],
            dependencies=[Depends(self.check_access_load)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-load-damage-list/",
            name=f"Load register of damaged parts from excel-file (as the user requested)",
            description=f'Loading register of damaged parts in "{self.table_name}" from excel-file (as the user requested).',
            endpoint=self.endpoint_load_damage_list,
            methods=["PUT"],
            dependencies=[Depends(self.check_access_load)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-export-special/",
            name=f"Export records to excel-file (as the user requested).",
            description=f'Exporting records from "{self.table_name}" to excel-file (as the user requested).',
            endpoint=self.endpoint_export_special,
            methods=["PUT"],
            dependencies=[Depends(self.check_access_export)],
        )

    async def endpoint_load_special(
        self,
        background_tasks: BackgroundTasks,
        source_type: SourceType,
        uploaded_file: UploadFile = File(...),
        is_overwrite: bool = False,
        is_async: bool = False,
        db: Session = Depends(get_db),
        engine: Engine = Depends(get_engine),
        engine_ora: Engine = Depends(get_engine_ora),
    ) -> Any:
        # checking for valid file types
        if not is_excel_by_content_type(uploaded_file.content_type):
            raise HTTPException(
                status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
            )
        username = PARSED_CONFIG.username
        parent_name = sys._getframe().f_code.co_name  # function_name
        parent_id = -random.randint(1, 10000)  # send ID<0 for get new ID such as ID
        log_db = LogDB(db)
        log_db.put(
            parent_id=parent_id,
            parent_name=parent_name,
            type_log=MyLogTypeEnum.START,
            msg=f'Старт функции "{parent_name}" (источник данных - "{source_type.value}") '
            f'для обработки "{uploaded_file.filename}"',
        )
        args = (
            self.load_special,
            db,
            engine,
            engine_ora,
            source_type,
            uploaded_file,
            is_overwrite,
            username,
            log_db,
            is_async,
        )
        if is_async:
            background_tasks.add_task(*args)
            result = f'Функция "{parent_name}" для обработки "{uploaded_file.filename}" запущена в фоновом режиме'
        else:
            loop = asyncio.get_running_loop()
            result = await asyncio.gather(loop.run_in_executor(None, *args))
        result = {"message": result[0] if type(result) is list else result, "log_id": log_db.id}
        write_user_history(
            db=db,
            username=username,
            message=f'Функцией "{parent_name}" (источник данных - "{source_type.value}") '
            f'обработан "{uploaded_file.filename}" ({result})',
        )
        return result

    async def endpoint_load_damage_list(
        self,
        uploaded_file: UploadFile = File(...),
        is_overwrite: bool = False,
        db: Session = Depends(get_db),
    ) -> Any:
        # checking for valid file types
        if not is_excel_by_content_type(uploaded_file.content_type):
            raise HTTPException(
                status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
            )
        username = PARSED_CONFIG.username
        parent_name = sys._getframe().f_code.co_name  # function_name
        parent_id = -random.randint(1, 10000)  # send ID<0 for get new ID such as ID
        log_db = LogDB(db)
        log_db.put(
            parent_id=parent_id,
            parent_name=parent_name,
            type_log=MyLogTypeEnum.START,
            msg=f'Старт функции "{parent_name}" для обработки "{uploaded_file.filename}"',
        )
        result = self.load_damage_list(db, uploaded_file, is_overwrite, username, log_db)

        write_user_history(
            db=db,
            username=username,
            message=f'Функцией "{parent_name}" обработан "{uploaded_file.filename}" ({result["msg"]})',
        )
        file_result = table_writer(dataframes={"finding_length": result["error_df"]}, param="xlsx")
        response = StreamingResponse(iter([file_result.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
        file_name = f"damage_list_error_{date.today()}.xlsx"
        response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
        response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        response.headers["message"] = transliteration(result["msg"])
        return response

    async def endpoint_export_special(
        self,
        db: Session = Depends(get_db),
        filter_model: Optional[dict] = None,
        sort_model: Optional[list[dict]] = None,
        skip: int = 0,
        limit: int = 99999,
    ):
        file_result = await self.export_special(db, filter_model, sort_model, skip=skip, limit=limit)
        response = StreamingResponse(iter([file_result.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
        file_name = f"{self.table_name}_{date.today()}.xlsx"
        response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
        response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return response

    def load_special(
        self,
        db: Session,
        engine: Engine,
        engine_ora: Engine,
        source_type: SourceType,
        uploaded_file: UploadFile,
        is_overwrite=True,
        username: str = "",
        log_db: LogDB = None,
        is_background: bool = False,
    ):
        if source_type == SourceType.SAP:
            LOADCOLS = [
                "Документ",
                "Описание системы",
                "Описание операции",
                "Внешний номер договора",
                "Номер вагона",
                "Наименование филиала",
                "Дата проводки",
                "Завод",
                "Номер детали производителя",
                "Год",
                "Обозначение склада",
                "ОбодЛ",
                "ОбодП",
                "Вид дефекта номерной детали",
                "Цена",
                "Описание типа крепления",
            ]
            CONVERTERS = {
                "Завод": str,
                "Номер вагона": int,
                "Номер детали производителя": str,
                "Год": str,
            }
            MAPPING = {
                "Документ": "document_num",
                "Описание системы": "system",
                "Внешний номер договора": "external_contract_num",
                "Номер вагона": "wagon_num",
                "Наименование филиала": "branch_name",
                "Дата проводки": "date_detect",
                "Обозначение склада": "storage_name",
                "ОбодЛ": "thickness_left_rim",
                "ОбодП": "thickness_right_rim",
                "Вид дефекта номерной детали": "rejection_code",
                "Цена": "part_cost",
                "Описание типа крепления": "mounting_type",
            }
            SAVECOLS = ["date_upload", "load_from", "part_number", "railway"]
            is_cumulative_headlines = False
        elif source_type == SourceType.VAREKS:
            LOADCOLS = [
                "Вагон",
                "Дата",
                "Сняли",
                "ОбодДо",
                "Причина",
                "Цена",
                "Депо",
            ]
            CONVERTERS = dict()
            MAPPING = {
                "Вагон": "wagon_num",
                "Дата": "date_detect",
                "Сняли": "part_number",
                "Причина": "rejection_code",
                "Цена": "part_cost",
                "Депо": "storage_name",
            }
            SAVECOLS = [
                "date_upload",
                "load_from",
                "thickness_left_rim",
                "thickness_right_rim",
                "branch_name",
                "railway",
                "mounting_type",
            ]
            is_cumulative_headlines = False
        elif source_type == SourceType.ASU_VRK:
            LOADCOLS = [
                "№ вагона",
                "Дата окончания ремонта",
                "Дорога",
                "ВЧДР",
                "Снято Номер детали",
                "Завод",
                "Год",
                "Характеристика",
                "Толщина обода",
                "Стоимость детали",
                "Вид дефекта и его размер",
            ]
            CONVERTERS = {
                "Завод": str,
                "Снято Номер детали": str,
                "Год": str,
            }
            MAPPING = {
                "№ вагона": "wagon_num",
                "Дата окончания ремонта": "date_detect",
                "Дорога": "railway",
                "ВЧДР": "storage_name",
                "Характеристика": "mounting_type",
                "Толщина обода": "thickness_left_rim",
                "Стоимость детали": "part_cost",
                "Вид дефекта и его размер": "rejection_code",
            }
            SAVECOLS = [
                "date_upload",
                "load_from",
                "part_number",
                "thickness_right_rim",
                "branch_name",
            ]
            is_cumulative_headlines = True
        elif source_type == SourceType.DAMAGE_LIST:
            LOADCOLS = [
                "Вид повреждения",
                "№ вагона",
                "Дата начала ремонта ВУ-23",
                "Дата выхода из ремонта ВУ-23",
                # 1
                "Наименование Контрагента по возмещению 1",
                "Дата претензии 1",
                "Внутренний № претензии 1",
                "Внешний № претензии 1",
                "Дата возмещение 1",
                "Сумма претензии 1",
                "Полученная сумма ущерба 1",
                "Возмещено страховой компанией 1",
                # 2
                "Наименование Контрагента по возмещению 2",
                "Дата претензии 2",
                "Внутренний № претензии 2",
                "Внешний № претензии 2",
                "Дата возмещение 2",
                "Сумма претензии 2",
                "Полученная сумма ущерба 2",
                "Возмещено страховой компанией 2",
                # 3
                "Наименование Контрагента по возмещению 3",
                "Дата претензии 3",
                "Внутренний № претензии 3",
                "Внешний № претензии 3",
                "Дата возмещение 3",
                "Сумма претензии 3",
                "Полученная сумма ущерба 3",
                "Возмещено страховой компанией 3",
                # sum
                "ИТОГО Сумма претензии",
                "ИТОГО Полученная сумма ущерба",
                "ИТОГО Возмещено страховой компанией",
            ]
            CONVERTERS = {
                "№ вагона": int,
                "Внешний № претензии 1": str,
                "Внешний № претензии 2": str,
                "Внешний № претензии 3": str,
            }
            MAPPING = {
                # 1
                "Наименование Контрагента по возмещению 1": "claim_refund_partner_1",
                "Дата претензии 1": "claim_date_1",
                "Внутренний № претензии 1": "claim_internal_number_1",
                "Внешний № претензии 1": "claim_external_number_1",
                "Дата возмещение 1": "claim_refund_date_1",
                "Сумма претензии 1": "claim_sum_1",
                "Полученная сумма ущерба 1": "claim_sum_damage_1",
                "Возмещено страховой компанией 1": "claim_partner_payed_1",
                # 2
                "Наименование Контрагента по возмещению 2": "claim_refund_partner_2",
                "Дата претензии 2": "claim_date_2",
                "Внутренний № претензии 2": "claim_internal_number_2",
                "Внешний № претензии 2": "claim_external_number_2",
                "Дата возмещение 2": "claim_refund_date_2",
                "Сумма претензии 2": "claim_sum_2",
                "Полученная сумма ущерба 2": "claim_sum_damage_2",
                "Возмещено страховой компанией 2": "claim_partner_payed_2",
                # 3
                "Наименование Контрагента по возмещению 3": "claim_refund_partner_3",
                "Дата претензии 3": "claim_date_3",
                "Внутренний № претензии 3": "claim_internal_number_3",
                "Внешний № претензии 3": "claim_external_number_3",
                "Дата возмещение 3": "claim_refund_date_3",
                "Сумма претензии 3": "claim_sum_3",
                "Полученная сумма ущерба 3": "claim_sum_damage_3",
                "Возмещено страховой компанией 3": "claim_partner_payed_3",
                # sum
                "ИТОГО Сумма претензии": "claim_sum_all",
                "ИТОГО Полученная сумма ущерба": "claim_sum_damage_all",
                "ИТОГО Возмещено страховой компанией": "claim_partner_payed_all",
            }
            SAVECOLS = []
            is_cumulative_headlines = False

        time_start = datetime.now()
        content = uploaded_file.file.read()  # async read
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = read_excel_with_find_headers(
                    file=content,
                    headers_list=LOADCOLS,
                    converters=CONVERTERS,
                    is_cumulative_headlines=is_cumulative_headlines,
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
        report_df.rename(columns=MAPPING, inplace=True)

        # filter by the conditions in the technical specification
        if source_type == SourceType.SAP:
            report_df = report_df.loc[(report_df["Описание операции"] != "Установка")]
        if source_type != SourceType.DAMAGE_LIST:
            report_df = self._filtered_by_rejection_code(db, report_df)
        log_db.add(msg=f'После фильтрации, по "Причина" (вид дефекта), осталось {report_df.shape[0]} строк.')

        if source_type == SourceType.SAP:
            report_df["part_number"] = (
                report_df["Завод"].fillna(0).astype(int).map(str)
                + "-"
                + report_df["Номер детали производителя"].fillna(0).astype(int).map(str)
                + "-"
                + report_df["Год"].fillna(0).astype(int).map(str)
            )
            report_df["load_from"] = "SAP"
            report_df["system"] = report_df["system"].str.slice(0, 20)
        elif source_type == SourceType.VAREKS:
            report_df["date_detect"] = report_df["date_detect"].map(lambda x: parse(x, dayfirst=True).date())
            report_df["thickness_left_rim"] = pd.to_numeric(
                report_df["ОбодДо"].map(lambda x: str(x).lstrip(" Лл")).str.extract(r"\b(\d+)\b")[0], errors="coerce"
            )
            report_df["thickness_right_rim"] = pd.to_numeric(
                report_df["ОбодДо"].map(lambda x: str(x)[3:].lstrip(" Пп")).str.extract(r"\b(\d+)\b")[0],
                errors="coerce",
            )
            report_df["mounting_type"] = np.NaN
            report_df["load_from"] = "Варекс"
        elif source_type == SourceType.ASU_VRK:
            report_df["thickness_right_rim"] = report_df["thickness_left_rim"]
            report_df["part_number"] = (
                report_df["Завод"].fillna(0).astype(int).map(str)
                + "-"
                + report_df["Снято Номер детали"].fillna(0).astype(int).map(str)
                + "-"
                + report_df["Год"].fillna(0).astype(int).map(str)
            )
            report_df["load_from"] = "АСУ ВРК"
        elif source_type == SourceType.DAMAGE_LIST:
            report_df = report_df.loc[
                (report_df["Вид повреждения"] == "6 - Повреждение колесных пар")
                | (report_df["Вид повреждения"] == "5 - Хищение/разоборудование")
            ]
            report_df["claim_external_number_1"].fillna("", inplace=True)
            report_df["claim_external_number_2"].fillna("", inplace=True)
            report_df["claim_external_number_3"].fillna("", inplace=True)

            amount_changed = 0
            error_df = DataFrame()
            for index, row in report_df.iterrows():
                wheel_set = db.query(models.WheelSet).filter(models.WheelSet.wagon_num == row["№ вагона"]).first()
                # wheel_set = (
                #     db.query(models.WheelSet)
                #     .filter(models.WheelSet.wagon_num == row["№ вагона"])
                #     .filter(
                #         models.WheelSet.date_detect.between(
                #             row["Дата начала ремонта ВУ-23"], row["Дата выхода из ремонта ВУ-23"]
                #         )
                #     )
                #     .first()
                # )
                if wheel_set:
                    if is_overwrite or not (
                        wheel_set.claim_date_1
                        or wheel_set.claim_refund_date_1
                        or wheel_set.claim_internal_number_1
                        or wheel_set.claim_external_number_1
                        or wheel_set.claim_sum_1
                        or wheel_set.claim_sum_damage_1
                        or wheel_set.claim_date_2
                        or wheel_set.claim_refund_date_2
                        or wheel_set.claim_internal_number_2
                        or wheel_set.claim_external_number_2
                        or wheel_set.claim_sum_2
                        or wheel_set.claim_sum_damage_2
                        or wheel_set.claim_date_3
                        or wheel_set.claim_refund_date_3
                        or wheel_set.claim_internal_number_3
                        or wheel_set.claim_external_number_3
                        or wheel_set.claim_sum_3
                        or wheel_set.claim_sum_damage_3
                    ):
                        if pd.isna(row["claim_refund_partner_1"]) and row["claim_external_number_1"].startswith(
                            "0524-"
                        ):
                            row["claim_refund_partner_1"] = 'СПАО "ИНГОССТРАХ"'
                        if pd.isna(row["claim_refund_partner_2"]) and row["claim_external_number_2"].startswith(
                            "0524-"
                        ):
                            row["claim_refund_partner_2"] = 'СПАО "ИНГОССТРАХ"'
                        if pd.isna(row["claim_refund_partner_3"]) and row["claim_external_number_3"].startswith(
                            "0524-"
                        ):
                            row["claim_refund_partner_3"] = 'СПАО "ИНГОССТРАХ"'
                        result_dict = {
                            key: val for key, val in row.to_dict().items() if key in MAPPING.values() and pd.notna(val)
                        }
                        db.query(models.WheelSet).filter(models.WheelSet.id == wheel_set.id).update(result_dict)
                        db.commit()
                        amount_changed += 1
                else:
                    error_df = pd.concat([error_df, DataFrame([row])])
            message = (
                f'В результате обработки "{uploaded_file.filename}" было изменено {amount_changed} строк.'
                f' (период выполнения - {str(datetime.now() - time_start).split(".", 2)[0]}).'
            )
            if error_df.shape[0] > 0:
                error_df.rename(columns={val: key for key, val in MAPPING.items()}, inplace=True)
                message += f"\nНе смогли загрузить следующие строки:\n{error_df}"
            log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
            return message

        self._add_branch_railway_mounting_type_cost(db, engine_ora, report_df)

        report_df = self._filtered_by_mounting_type(db, report_df)
        log_db.add(msg=f'После фильтрации, по "тип крепления", осталось {report_df.shape[0]} строк.')

        report_df["part_cost"] = report_df["part_cost"].fillna(0).astype(float)
        report_df["thickness_left_rim"] = report_df["thickness_left_rim"].fillna(0).astype(int)
        report_df["thickness_right_rim"] = report_df["thickness_right_rim"].fillna(0).astype(int)
        report_df["date_upload"] = date.today()

        # Del old records if 'is_overwrite' = True
        amount_delete = self._delete_old(db, report_df, is_overwrite)

        # Add new row in WheelSet
        amount_add = save_df_with_unique(
            db,
            engine,
            "wheel_set",
            report_df,
            unique_cols=["part_number", "date_detect"],
            cols=list(MAPPING.values()) + SAVECOLS,
        )
        amount_delete = f", удалено - {amount_delete} записей" if amount_delete else ""
        message = (
            f'В результате обработки "{uploaded_file.filename}" было {amount_add}'
            f'{amount_delete} (период выполнения - {str(datetime.now() - time_start).split(".", 2)[0]}).'
        )
        log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
        return message

    @staticmethod
    def load_damage_list(
        db: Session,
        uploaded_file: UploadFile,
        is_overwrite=True,
        username: str = "",
        log_db: LogDB = None,
    ):
        LOADCOLS = [
            "Вид повреждения",
            "№ вагона",
            "Дата начала ремонта ВУ-23",
            "Дата выхода из ремонта ВУ-23",
            # 1
            "Наименование Контрагента по возмещению 1",
            "Дата претензии 1",
            "Внутренний № претензии 1",
            "Внешний № претензии 1",
            "Дата возмещение 1",
            "Сумма претензии 1",
            "Полученная сумма ущерба 1",
            "Возмещено страховой компанией 1",
            # 2
            "Наименование Контрагента по возмещению 2",
            "Дата претензии 2",
            "Внутренний № претензии 2",
            "Внешний № претензии 2",
            "Дата возмещение 2",
            "Сумма претензии 2",
            "Полученная сумма ущерба 2",
            "Возмещено страховой компанией 2",
            # 3
            "Наименование Контрагента по возмещению 3",
            "Дата претензии 3",
            "Внутренний № претензии 3",
            "Внешний № претензии 3",
            "Дата возмещение 3",
            "Сумма претензии 3",
            "Полученная сумма ущерба 3",
            "Возмещено страховой компанией 3",
            # sum
            "ИТОГО Сумма претензии",
            "ИТОГО Полученная сумма ущерба",
            "ИТОГО Возмещено страховой компанией",
        ]
        CONVERTERS = {
            "№ вагона": int,
            "Внешний № претензии 1": str,
            "Внешний № претензии 2": str,
            "Внешний № претензии 3": str,
        }
        MAPPING = {
            # 1
            "Наименование Контрагента по возмещению 1": "claim_refund_partner_1",
            "Дата претензии 1": "claim_date_1",
            "Внутренний № претензии 1": "claim_internal_number_1",
            "Внешний № претензии 1": "claim_external_number_1",
            "Дата возмещение 1": "claim_refund_date_1",
            "Сумма претензии 1": "claim_sum_1",
            "Полученная сумма ущерба 1": "claim_sum_damage_1",
            "Возмещено страховой компанией 1": "claim_partner_payed_1",
            # 2
            "Наименование Контрагента по возмещению 2": "claim_refund_partner_2",
            "Дата претензии 2": "claim_date_2",
            "Внутренний № претензии 2": "claim_internal_number_2",
            "Внешний № претензии 2": "claim_external_number_2",
            "Дата возмещение 2": "claim_refund_date_2",
            "Сумма претензии 2": "claim_sum_2",
            "Полученная сумма ущерба 2": "claim_sum_damage_2",
            "Возмещено страховой компанией 2": "claim_partner_payed_2",
            # 3
            "Наименование Контрагента по возмещению 3": "claim_refund_partner_3",
            "Дата претензии 3": "claim_date_3",
            "Внутренний № претензии 3": "claim_internal_number_3",
            "Внешний № претензии 3": "claim_external_number_3",
            "Дата возмещение 3": "claim_refund_date_3",
            "Сумма претензии 3": "claim_sum_3",
            "Полученная сумма ущерба 3": "claim_sum_damage_3",
            "Возмещено страховой компанией 3": "claim_partner_payed_3",
            # sum
            "ИТОГО Сумма претензии": "claim_sum_all",
            "ИТОГО Полученная сумма ущерба": "claim_sum_damage_all",
            "ИТОГО Возмещено страховой компанией": "claim_partner_payed_all",
        }
        SAVECOLS = []
        is_cumulative_headlines = False

        time_start = datetime.now()
        content = uploaded_file.file.read()  # async read
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = read_excel_with_find_headers(
                    file=content,
                    headers_list=LOADCOLS,
                    converters=CONVERTERS,
                    is_cumulative_headlines=is_cumulative_headlines,
                )
            except Exception as e:
                msg = e.detail if type(e) is HTTPException else f"Некорректный формат файла - {str(e)}"
                log_db.add(msg=msg, type_log=MyLogTypeEnum.ERROR)
                raise HTTPException(status_code=422, detail=msg + f' (файл: "{uploaded_file.filename}")')

        report_df.columns = report_df.columns.str.replace("\n", " ").str.strip()
        log_db.add(msg=f"Из excel-файла считано {report_df.shape[0]} строк.", username=username)

        # removing unnecessary spaces
        report_df = report_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        report_df.rename(columns=MAPPING, inplace=True)

        # filter by the conditions in the technical specification
        report_df = report_df.loc[
            (report_df["Вид повреждения"] == "6 - Повреждение колесных пар")
            | (report_df["Вид повреждения"] == "5 - Хищение/разоборудование")
        ]
        report_df["claim_external_number_1"].fillna("", inplace=True)
        report_df["claim_external_number_2"].fillna("", inplace=True)
        report_df["claim_external_number_3"].fillna("", inplace=True)
        # report_df["Дата начала ремонта ВУ-23"].fillna(report_df["Дата выхода из ремонта ВУ-23"], inplace=True)
        # report_df["Дата выхода из ремонта ВУ-23"].fillna(report_df["Дата начала ремонта ВУ-23"], inplace=True)
        amount_changed = 0
        error_df = DataFrame()
        for index, row in report_df.iterrows():
            # wheel_set = (
            #     db.query(models.WheelSet)
            #     .filter(models.WheelSet.wagon_num == row["№ вагона"])
            #     .filter(
            #         models.WheelSet.date_detect.between(
            #             row["Дата начала ремонта ВУ-23"] - timedelta(days=90), row["Дата выхода из ремонта ВУ-23"]
            #         )
            #     )
            #     .first()
            # )
            wheel_set = db.query(models.WheelSet).filter(models.WheelSet.wagon_num == row["№ вагона"]).first()
            if wheel_set:
                if is_overwrite or not (
                    wheel_set.claim_date_1
                    or wheel_set.claim_refund_date_1
                    or wheel_set.claim_internal_number_1
                    or wheel_set.claim_external_number_1
                    or wheel_set.claim_sum_1
                    or wheel_set.claim_sum_damage_1
                    or wheel_set.claim_date_2
                    or wheel_set.claim_refund_date_2
                    or wheel_set.claim_internal_number_2
                    or wheel_set.claim_external_number_2
                    or wheel_set.claim_sum_2
                    or wheel_set.claim_sum_damage_2
                    or wheel_set.claim_date_3
                    or wheel_set.claim_refund_date_3
                    or wheel_set.claim_internal_number_3
                    or wheel_set.claim_external_number_3
                    or wheel_set.claim_sum_3
                    or wheel_set.claim_sum_damage_3
                ):
                    if pd.isna(row["claim_refund_partner_1"]) and row["claim_external_number_1"].startswith("0524-"):
                        row["claim_refund_partner_1"] = 'СПАО "ИНГОССТРАХ"'
                    if pd.isna(row["claim_refund_partner_2"]) and row["claim_external_number_2"].startswith("0524-"):
                        row["claim_refund_partner_2"] = 'СПАО "ИНГОССТРАХ"'
                    if pd.isna(row["claim_refund_partner_3"]) and row["claim_external_number_3"].startswith("0524-"):
                        row["claim_refund_partner_3"] = 'СПАО "ИНГОССТРАХ"'
                    result_dict = {
                        key: val for key, val in row.to_dict().items() if key in MAPPING.values() and pd.notna(val)
                    }
                    db.query(models.WheelSet).filter(models.WheelSet.id == wheel_set.id).update(result_dict)
                    db.commit()
                    amount_changed += 1
            else:
                error_df = pd.concat([error_df, DataFrame([row])])
        message = (
            f'В результате обработки "{uploaded_file.filename}" было изменено {amount_changed} строк.'
            f' (период выполнения - {str(datetime.now() - time_start).split(".", 2)[0]}).'
        )
        if error_df.shape[0] > 0:
            error_df.rename(columns={val: key for key, val in MAPPING.items()}, inplace=True)
            message += f" Не смогли загрузить {error_df.shape[0]} строк."
        log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH, username=username)
        return {"msg": message, "error_df": error_df}

    @staticmethod
    def _filtered_by_rejection_code(db: Session, df: DataFrame) -> DataFrame:
        # filter by rejection_code
        defect_type_filter = db.query(models.WheelSetFilter).all()
        df["is_filtered"] = np.NAN
        for defect_type in defect_type_filter:
            df.loc[
                df["rejection_code"].str.contains(defect_type.name, case=False, regex=False, na=False), "is_filtered"
            ] = True
        df = df.loc[df["is_filtered"].fillna(False)]
        return df

    @staticmethod
    def _filtered_by_mounting_type(db: Session, df: DataFrame) -> DataFrame:
        # filter by the mounting_type
        mounting_type_for_load = get_mounting_type_for_load(db)
        df = df.loc[df["mounting_type"].isin(mounting_type_for_load) | df["mounting_type"].isnull()]
        return df

    @staticmethod
    def _delete_old(db: Session, df: DataFrame, is_overwrite: bool = True) -> int:
        # Del old records if 'is_overwrite' = True
        amount_delete = 0
        if is_overwrite:
            for i, el in df.iterrows():
                amount_delete += (
                    db.query(models.WheelSet)
                    .filter(
                        (models.WheelSet.part_number == el["part_number"]),
                        (models.WheelSet.date_detect == el["date_detect"]),
                    )
                    .delete(synchronize_session="fetch")
                )
                db.commit()
        return amount_delete

    def _add_branch_railway_mounting_type_cost(self, db: Session, engine_ora: Engine, df: DataFrame):
        if df.empty:
            return
        for el in ("branch_name", "storage_name", "railway"):
            if el not in df.columns:
                df[el] = np.NAN

        mounting_type_all = get_mounting_type_all(db)
        for index, el in df.iterrows():
            if pd.notnull(el["storage_name"]) and (pd.isnull(el["branch_name"]) or pd.isnull(el["railway"])):
                result = (
                    db.query(models.Storage)
                    .filter(func.lower(models.Storage.name) == el["storage_name"].lower())
                    .first()
                )
                if result:
                    df.loc[index, "branch_name"] = el["branch_name"] if pd.notnull(el["branch_name"]) else result.branch
                    df.loc[index, "railway"] = el["railway"] if pd.notnull(el["railway"]) else result.railway

            if pd.isnull(el["mounting_type"]) and pd.notnull(el["rejection_code"]):
                # all_words = re.findall(r"[\w']+", el["rejection_code"].upper() )
                # result = [col for col in mounting_type_all if el["rejection_code"].upper().find(col) != -1]
                all_words = list(filter(None, re.split(r"[ .,;\!?:]+", el["rejection_code"].upper())))
                result = [col for col in mounting_type_all if col in all_words]
                if result:
                    df.loc[index, "mounting_type"] = max(result, key=len)
                    el["mounting_type"] = max(result, key=len)

            if (
                pd.notnull(el["mounting_type"])
                and (pd.isnull(el["part_cost"]) or el["part_cost"] == 0)
                and (pd.notnull(el["thickness_left_rim"]) or pd.notnull(el["thickness_right_rim"]))
            ):
                result = (
                    db.query(models.MountingTypeMap)
                    .filter(func.upper(models.MountingTypeMap.name_from) == el["mounting_type"].upper())
                    .first()
                )
                mounting_type = result.name_to if result else el["mounting_type"]
                result = (
                    db.query(models.WheelSetCost)
                    .filter(func.upper(models.WheelSetCost.mounting_type) == mounting_type.upper())
                    .all()
                )
                rim_thickness_min = min(el["thickness_left_rim"], el["thickness_right_rim"])
                for cost in result:
                    if (
                        cost.rim_thickness_min <= rim_thickness_min <= cost.rim_thickness_max
                        or cost.rim_thickness_min >= rim_thickness_min >= cost.rim_thickness_max
                    ):
                        df.loc[index, "part_cost"] = cost.cost
                        break

        self._get_branch_railway_from_oracle(engine_ora, df)
        return

    @staticmethod
    def _get_branch_railway_from_oracle(engine_ora: Engine, df: DataFrame):
        def remove_abbreviation(name: str):
            name = name.upper()
            abbreviation = ("ВЧДР", "ВРЗ", "ВРД", "ВУ ", "ООО", "ДИ ", "ВРП", "ВКМ", "НВК", "БВРП", "РВД ")
            for el in abbreviation:
                name = name.replace(el, "")
            name = name.lstrip("-").rstrip("-").rstrip(".")
            # replace \n to space and remove double-space
            return " ".join(str(name).split())

        if df.empty:
            return
        for index, el in df.iterrows():
            if pd.notnull(el["storage_name"]) and (pd.isnull(el["branch_name"]) or pd.isnull(el["railway"])):
                storage_name = remove_abbreviation(el["storage_name"])
                result_df = pd.read_sql(
                    f"""
                        SELECT
                            s.ST_CODE, s.ST_NAME
                            , vr.RW_CODE, vr.RW_SHORT_NAME, vr.RW_NAME
                            , f.ORG_ID, f.SHORTNAME org_shortname, f.NAME org_name
                        FROM ssp.STATIONS s
                        INNER JOIN nsi.V_RAILWAY_SYSDATE vr
                            ON s.ROADID = vr.RW_CODE
                        INNER JOIN ssp.ORG_FILIAL f
                            ON s.BRANCH_ID = f.ORG_ID AND vr.RW_CODE = f.RW_CODE
                        WHERE  s.ST_NAME LIKE '%{storage_name}%'
                    """,
                    con=engine_ora,
                )
                if not result_df.empty:
                    result_df["distance"] = result_df["st_name"].apply(Levenshtein.distance, s2=storage_name)
                    index_ora = result_df["distance"].idxmin()
                    df.loc[index, "branch_name"] = (
                        el["branch_name"]
                        if pd.notnull(el["branch_name"])
                        else result_df.loc[index_ora, "org_shortname"]
                    )
                    df.loc[index, "railway"] = (
                        el["railway"] if pd.notnull(el["railway"]) else result_df.loc[index_ora, "rw_name"]
                    )
        return

    async def export_special(self, db: Session, *args, **kwargs):
        MAPPING = {
            "load_from": "Источник данных",
            "wagon_num": "Номер вагона",
            "branch_name": "Филиал",
            "railway": "Дорога",
            "date_detect": "Дата выявления хищения",
            "part_number": "Номер похищенной детали",
            "thickness_left_rim": "Толщина левого обода, мм",
            "thickness_right_rim": "Толщина правого обода, мм",
            "rejection_code": "Код браковки (причина)",
            "part_cost": "Стоимость похищенной детали, руб.",
            "mounting_type": "Описание типа крепления",
            "system": "Описание системы",
            "storage_name": "Обозначение склада",
            "external_contract_num": "Внешний номер договора",
            "document_num": "Документ",
            "repay": "Всего возмещено, руб.",
            "repay_author": "Восстановление вагона за счёт ВРП/виновником, руб.",
            "insurance_notification_date": "Дата направления уведомления в СК",
            "insurance_notification_num": "Исходящий номер уведомления в СК",
            "insurance_name": "Наименование СК",
            "is_insurance_of_carrier": "Страховая компания перевозчика (да/нет)",
            "insurance_number": "Номер убытка (присваивается СК)",
            "insurance_claim_number": "Номер претензии",
            "insurance_payment_total": "Общий размер требований о выплате СК, руб.",
            "insurance_payment_date": "Дата выплаты страхового возмещения",
            "insurance_payment_done": "Выплаченная сумма СК, руб.",
            "author_pretension_date": "Дата направления претензии виновному",
            "author_name": "Наименование виновника",
            "author_pretension_number": "Исходящий номер претензии",
            "author_payment_total": "Общий размер требований о выплате, руб.",
            "author_payment_date": "Дата выплаты виновником по претензии",
            "author_payment_done": "Выплаченная сумма, руб.",
            "author_lawyer_date": "Дата передачи материала Юристам",
            "author_lawyer_number": "Номер Арбитражного дела",
            "police_date": "Дата передачи материала БР",
            "police_ovd_date": "Дата направления заявления в ОВД",
            "police_ovd_name": "Наименование ОВД",
            "police_payment": "Размер ущерба, руб. ",
            "police_decision": "Процессуальное решение по заявлению (возбуждение уг. дела / отказ в ВУД)",
            "police_decision_date": "Дата процессуального решения",
            # 1
            "claim_refund_partner_1": "Наименование Контрагента по возмещению 1",
            "claim_date_1": "Дата претензии 1",
            "claim_internal_number_1": "Внутренний № претензии 1",
            "claim_external_number_1": "Внешний № претензии 1",
            "claim_refund_date_1": "Дата возмещения 1",
            "claim_sum_1": "Сумма претензии 1, руб.",
            "claim_sum_damage_1": "Полученная сумма ущерба 1, руб.",
            "claim_partner_payed_1": "Возмещено страховой компанией 1, руб.",
            # 2
            "claim_refund_partner_2": "Наименование Контрагента по возмещению 2",
            "claim_date_2": "Дата претензии 2",
            "claim_internal_number_2": "Внутренний № претензии 2",
            "claim_external_number_2": "Внешний № претензии 2",
            "claim_refund_date_2": "Дата возмещения 2",
            "claim_sum_2": "Сумма претензии 2, руб.",
            "claim_sum_damage_2": "Полученная сумма ущерба 2, руб.",
            "claim_partner_payed_2": "Возмещено страховой компанией 2, руб.",
            # 3
            "claim_refund_partner_3": "Наименование Контрагента по возмещению 3",
            "claim_date_3": "Дата претензии 3",
            "claim_internal_number_3": "Внутренний № претензии 3",
            "claim_external_number_3": "Внешний № претензии 3",
            "claim_refund_date_3": "Дата возмещения 3",
            "claim_sum_3": "Сумма претензии 3, руб.",
            "claim_sum_damage_3": "Полученная сумма ущерба 3, руб.",
            "claim_partner_payed_3": "Возмещено страховой компанией 3, руб.",
            # sum
            "claim_sum_all": "Сумма претензии, итого, руб.",
            "claim_sum_damage_all": "Полученная сумма ущерба, итого, руб.",
            "claim_partner_payed_all": "Возмещено страховой компанией, итого, руб.",
        }

        result = await self.get_list_advanced(db, *args, **kwargs)
        df = self._df_from_sql(result)

        # convert "True/False" to "да/нет"
        df = df_replace_true_false_with_yes_no(df, self.model)

        df.rename(columns=MAPPING, inplace=True)
        df = df[MAPPING.values()].sort_values(["Дата выявления хищения", "Дорога", "Филиал"])
        df["Дата выявления хищения"] = pd.to_datetime(df["Дата выявления хищения"], errors="coerce").dt.strftime(
            "%d.%m.%Y"
        )
        stream = table_writer(dataframes={"Поиск колесных пар": df}, param="xlsx")
        return stream


class StorageCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-load-special/",
            name=f"Load from excel-file (as the user requested)",
            description=f'Loading the records in "{self.table_name}" from excel-file (as the user requested).',
            endpoint=self.endpoint_load_special,
            methods=["POST"],
            dependencies=[Depends(self.check_access_load)],
        )

    @staticmethod
    def load_special(
        db: Session,
        engine: Engine,
        uploaded_file: UploadFile,
        is_overwrite=True,
        username: str = "",
        log_db: LogDB = None,
        is_background: bool = False,
    ):
        LOADCOLS = ["Обозначение склада/ВРП", "Дорога", "Филиал"]
        CONVERTERS = dict()
        MAPPING = {"Обозначение склада/ВРП": "name", "Дорога": "railway", "Филиал": "branch"}

        time_start = datetime.now()
        content = uploaded_file.file.read()  # async read
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
            try:
                report_df = read_excel_with_find_headers(file=content, headers_list=LOADCOLS, converters=CONVERTERS)
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
        report_df.rename(columns=MAPPING, inplace=True)

        # del empty records
        report_df = report_df.loc[
            report_df["name"].notnull() & (report_df["railway"].notnull() | report_df["branch"].notnull())
        ]

        # Del old records if 'is_overwrite' = True
        amount_delete = 0
        if is_overwrite:
            for i, el in report_df.iterrows():
                amount_delete += (
                    db.query(models.Storage)
                    .filter(models.Storage.name == el["name"])
                    .delete(synchronize_session="fetch")
                )
                db.commit()

        # Add new row in Storage
        amount_add = save_df_with_unique(
            db,
            engine,
            "storage",
            report_df,
            unique_cols=["name"],
            cols=list(MAPPING.values()),
        )
        amount_delete = f", удалено - {amount_delete} записей" if amount_delete else ""
        message = (
            f'В результате обработки "{uploaded_file.filename}" было {amount_add}'
            f'{amount_delete} (период выполнения - {str(datetime.now() - time_start).split(".", 2)[0]}).'
        )
        log_db.add(msg=message, type_log=MyLogTypeEnum.FINISH)
        return message


class MountingTypeCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            "/get-mounting-type-for-load/",
            response_model=list,
            name="Get all types of Mounting that we take for loading.",
            description="Get all types of Mounting that we take for loading.",
            endpoint=self.endpoint_get_mounting_type_for_load,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.router.add_api_route(
            "/get-mounting-type-for-cost/",
            response_model=list,
            name="Get all types of Mounting for WheelSetCost.",
            description='Get all the mounting types we have in the "MountingTypeMap" (name_to) table '
            'to fill in "WheelSetCost".',
            endpoint=self.endpoint_get_mounting_type_for_cost,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.router.add_api_route(
            "/get-mounting-type-all/",
            response_model=list,
            name="Get all types of Mounting.",
            description="Get all types of Mounting that we analyzed during loading. "
            '("mounting_type" + "mounting_type_map" + "wheel_set_cost")',
            endpoint=self.endpoint_get_mounting_type_all,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )

    async def endpoint_get_mounting_type_for_load(self, db: Session = Depends(get_db)) -> list:
        return get_mounting_type_for_load(db)

    async def endpoint_get_mounting_type_for_cost(self, db: Session = Depends(get_db)) -> list:
        return get_mounting_type_for_cost(db)

    async def endpoint_get_mounting_type_all(self, db: Session = Depends(get_db)) -> list:
        return get_mounting_type_all(db)


def get_mounting_type_all(db: Session) -> list:
    result = [el.name for el in db.query(models.MountingType).all()]
    result.extend([el.mounting_type for el in db.query(models.WheelSetCost).all()])
    for el in db.query(models.MountingTypeMap).all():
        result.extend([el.name_from, el.name_to])
    return sorted(list(set(map(str.upper, result))))  # remove duplicates


def get_mounting_type_for_cost(db: Session) -> list:
    result = [el.name_to for el in db.query(models.MountingTypeMap).all()]
    return sorted(list(set(map(str.upper, result))))  # remove duplicates


def get_mounting_type_for_load(db: Session) -> list:
    result = [el.name for el in db.query(models.MountingType).filter(models.MountingType.is_loading).all()]
    return sorted(list(set(map(str.upper, result))))  # remove duplicates
