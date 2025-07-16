import functools
from loguru import logger
import os
import re
import sys
import uuid
import warnings
import weakref
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum

from datetime import datetime, date as date_class, timedelta
from dateutil.parser import parse
from fastapi import status
from functools import wraps
from io import BytesIO, StringIO
from itertools import chain
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pathlib import Path
from time import process_time
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dateutil import parser as date_parser
from fastapi import HTTPException
from pandas import DataFrame, ExcelWriter, MultiIndex, Series
from pandas.core.dtypes.common import is_bool_dtype
from pydantic import root_validator, BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, DeclarativeMeta, InstanceState
from sqlalchemy.orm.collections import InstrumentedList
from yaml import SafeLoader

# from app.utils.utils_df import MAPPING_NAME


path_matcher = re.compile(r"\$\{([^}^{]+)\}")


def path_constructor(loader, node):
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return f"{os.environ.get(env_var)}{value[match.end():]}"


def read_yaml(path: Path) -> Mapping:
    yaml.add_implicit_resolver("!path", path_matcher, None, SafeLoader)
    yaml.add_constructor("!path", path_constructor, SafeLoader)

    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge(left: Mapping, right: Mapping) -> Mapping:
    """
    Merge two mappings objects together, combining overlapping Mappings,
    and favoring right-values
    left: The left Mapping object.
    right: The right (favored) Mapping object.
    NOTE: This is not commutative (merge(a,b) != merge(b,a)).
    """
    merged = {}

    left_keys = frozenset(left)
    right_keys = frozenset(right)

    # Items only in the left Mapping
    for key in left_keys - right_keys:
        merged[key] = left[key]

    # Items only in the right Mapping
    for key in right_keys - left_keys:
        merged[key] = right[key]

    # in both
    for key in left_keys & right_keys:
        left_value = left[key]
        right_value = right[key]

        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):  # recursive merge
            merged[key] = merge(left_value, right_value)
        else:  # overwrite with right value
            merged[key] = right_value

    return merged


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(process_time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(process_time() * 1000)) - start
            logger.info(f"Total execution time {func.__name__}: {end_ if end_ > 0 else 0} ms")

    return _time_it


def multi_case_filter_evaluation(filters: list[str], field: str):
    parsed_filters = []
    for item in filters:
        if isinstance(item, str):
            parsed_filters.append(item.lower())
        elif isinstance(item, Enum):
            parsed_filters.append(item.value.lower())
    if isinstance(field, str):
        if field.lower() in parsed_filters:
            return True
    elif isinstance(field, Enum):
        if field.value.lower() in parsed_filters:
            return True
    return False


def filter_parsed_models(filters: dict[str, list], sequence: list):
    filtered_sequence = []
    if not filters:
        return sequence
    for item in sequence:
        for k, v in filters.items():
            field = getattr(item, k)
            if multi_case_filter_evaluation(v, field):
                filtered_sequence.append(item)
    return list(set(filtered_sequence))


def table_writer(dataframes: dict[Optional[str], DataFrame], param: Optional = "xlsx") -> BytesIO:
    def excellent_header():
        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        # Add a header format.
        header_format = workbook.add_format(
            {"text_v_align": 2, "align": "center", "text_wrap": True, "bold": True, "fg_color": "#ffcccc", "border": 1}
        )
        if isinstance(dataframe.columns, MultiIndex):
            # multilevel header
            for row_num, value in enumerate(dataframe.columns.names):
                worksheet.write(row_num, 0, value, header_format)
        for col_num, value in enumerate(dataframe.columns.values):
            if isinstance(value, tuple):
                # multilevel header
                for level in range(len(value)):
                    worksheet.write(level, col_num + 1, value[level], header_format)
            else:
                worksheet.write(0, col_num, value, header_format)
            column_len = dataframe.iloc[:, col_num].astype(str).str.len().max()
            # Setting the length if the column header is larger than the max column value length (<= 30)
            column_len = min(max(column_len, len(value) // 2 + 1) + 3, 30)
            # set the column length
            worksheet.set_column(col_num, col_num, column_len)

    output = BytesIO()
    if param == "xlsx":
        max_row = 1000000
        writer = ExcelWriter(output, engine="xlsxwriter")
        for count, (name, dataframe) in enumerate(dataframes.items()):
            sheet_name_begin = name if name else f"sheet {count}"
            sheet_count = int((dataframe.shape[0] - 1) / max_row + 1)
            # replace dot with comma for Decimal
            dataframe = df_convert_number_to_number(dataframe)
            # dataframe = dataframe.copy().apply(pd.to_numeric, errors="ignore")
            for index in range(sheet_count):
                if sheet_count > 1:
                    sheet_name = f"{sheet_name_begin} ({index})"
                    df = dataframe.loc[index * max_row : (index + 1) * max_row - 1]
                else:
                    sheet_name = sheet_name_begin
                    df = dataframe
                if isinstance(df.columns, MultiIndex):
                    # multilevel header
                    df.to_excel(writer, sheet_name=sheet_name, index=True, freeze_panes=(2, 0))
                else:
                    df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 0))
                excellent_header()
            # writer.save()
        writer.close()
    elif param == "csv":
        for name, dataframe in dataframes.items():
            dataframe.to_csv(output, index=False)
            # output.seek(0)
    return output


def df_to_new_table(db: Session, engine: Engine, df, table_name, is_cast_uppercase=False):
    # def load_csv_into_new_table(file, db_schema, db_table, db_name='default',
    #                             is_cast_uppercase=False, delimeter='|', encoding='utf-8', decimal='.'):

    def _prepare_create_table_cmd(df):
        dtype_conv_dict = {
            "datetime64[ns]": "date",
            "int64": "bigint",  # some values exceed 4 bytes of pg integer...
            "float64": "numeric",
            "object": "varchar",
            "string": "varchar",
            "bool": "bool",
        }
        create_table_cmd = f'create table "{table_name}" (\n'
        for col in df.columns:
            dtype = str(df[col].dtype)
            field_mod_str = ""
            if dtype in ("object", "string"):
                max_len = df[col].str.len().max()
                max_len = 10 if max_len is np.nan else int(max_len + 10)
                field_mod_str = f"({max_len})"
            pg_type = dtype_conv_dict[dtype]
            create_table_cmd = create_table_cmd + f'"{col}" {pg_type}{field_mod_str},\n'
        create_table_cmd = create_table_cmd[:-2] + ");"
        return create_table_cmd

    print(f"Casting types ...")
    # try to cast all not determined typed to dates
    for col in df.select_dtypes("object").columns:
        df.loc[:, col] = pd.to_datetime(df.loc[:, col], errors="ignore")
    print("done!")

    print(f"Creating new table ...")
    comment_dict = {MAPPING_NAME[col]: col for col in df.columns if col in MAPPING_NAME_COGNOS}
    df.rename(columns=MAPPING_NAME, inplace=True)
    if is_cast_uppercase:
        df.columns = [f"{col.upper()}" for col in df.columns]
    # else:
    #     df.columns = [f'{col}' for col in df.columns]

    drop_table_cmd = f'drop table if exists "{table_name}" CASCADE;'
    db.execute(text(drop_table_cmd))
    db.commit()

    create_table_cmd = _prepare_create_table_cmd(df)
    db.execute(text(create_table_cmd))
    db.commit()

    for col, comment in comment_dict.items():
        create_comment_cmd = f"comment on column {table_name}.{col} is '{comment}';"
        db.execute(text(create_comment_cmd))
    db.commit()
    print("done!")

    print(f"Loading into db ...")
    save_df_to_model_via_csv(engine, df, cols=df.columns, db_table=table_name)
    print("done!")


def save_df_to_model_via_csv(engine: Engine, df, cols=None, model_class=None, db_table=None):
    # fastest way to insert into model raw data via CSV file
    assert model_class or db_table, "model_class or db_table should be provided"
    if df.empty:
        return
    cols = df.columns if cols is None else cols
    if model_class:
        db_table = model_class.__tablename__
        cols_from_model = set(map(lambda x: str(x).split(".")[-1], model_class.__table__.columns))
        cols = list(cols_from_model.intersection(cols))

        for col in cols:
            if str(model_class.__dict__[col].type).startswith("VARCHAR("):
                df.loc[df[col].isnull(), col] = ""
                # df[col].fillna("", inplace=True)
                max_len_db_col = model_class.__dict__[col].type.length
                max_len_df_col = max(df[col].apply(str).apply(len))
                if max_len_df_col > max_len_db_col:
                    df[col] = df[col].str.slice(0, max_len_db_col - 1)
                    print(f"{col} max len ={max_len_df_col}")

    output = StringIO()
    # import csv
    df[cols].to_csv(output, sep="\t", header=False, index=False, quotechar="&")  # , quoting=csv.QUOTE_MINIMAL)
    # df[cols].to_csv(output, sep="\t", header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    fake_conn = engine.raw_connection()
    fake_cur = fake_conn.cursor()
    fake_cur.copy_from(output, db_table, null="", columns=cols)
    fake_conn.commit()


def save_df_with_unique(
    db: Session,
    engine: Engine,
    db_table_name: str,
    df: pd.DataFrame,
    cols: list = None,
    unique_cols: list = None,
    is_update_exist: bool = False,
):
    if df.shape[0] == 0:
        return f"0 строк (возможно, причина в условиях фильтрации или исходной выборке)."
        # return {"add_row": 0, "update_row": 0, "message": "DataFrame is empty"}

    if cols is None:
        cols = df.columns if unique_cols is None else unique_cols
    # Removing 'cols' item if not in 'df.columns'
    for el in set(cols) - set(df.columns):
        cols.remove(el)
    unique_cols = deepcopy(unique_cols) if unique_cols else deepcopy(cols)
    if not isinstance(unique_cols[0], list):
        unique_cols = [unique_cols]

    # delete rows with non-unique keys
    for unique_cols_el in unique_cols:
        df.drop_duplicates(subset=unique_cols_el, keep="last", ignore_index=True, inplace=True)

    # create tmp table
    db_table_name_tmp = f"{db_table_name}_tmp_{str(uuid.uuid4()).replace('-', '_')[:5]}"
    df[cols].to_sql(db_table_name_tmp, con=engine, if_exists="replace", index=False)

    # updating records if required
    if is_update_exist:
        all_unique_cols = list(set(chain.from_iterable(unique_cols)))  # extract nested lists with remove duplicates
        sql_command = (
            f"UPDATE {db_table_name} as new "
            f"SET {', '.join([el + ' = tmp.' + el for el in cols])} "
            f"FROM {db_table_name_tmp} AS tmp "
            f"LEFT JOIN {db_table_name} AS cur "
            f"USING ({', '.join(all_unique_cols)}) "
            f"WHERE new.id = cur.id and not cur.id is null;"
        )
        update_rows = db.execute(text(sql_command)).rowcount
    else:
        update_rows = 0

    # calculate records that are not in the table for all variants of unique columns
    db_table_name_tmp_2 = f"{db_table_name_tmp}_"
    for unique_cols_el in unique_cols:
        # create empty table
        db.execute(text(f'CREATE TABLE "{db_table_name_tmp_2}" (like "{db_table_name_tmp}");'))
        sql_command = (
            f"INSERT INTO {db_table_name_tmp_2} ({', '.join(cols)}) "
            f"SELECT DISTINCT {', '.join(['tmp.' + el for el in cols])} "
            f"FROM {db_table_name_tmp} AS tmp "
            f"LEFT JOIN {db_table_name} AS cur "
            f"USING ({', '.join(unique_cols_el)}) WHERE cur.id is null;"
        )
        db.execute(text(sql_command)).rowcount
        db.execute(text(f"DROP TABLE {db_table_name_tmp} CASCADE;"))
        db.execute(text(f"ALTER TABLE {db_table_name_tmp_2} RENAME TO {db_table_name_tmp};"))
        db.commit()
    # add records that are not in the table for all variants of unique columns
    sql_command = (
        f"INSERT INTO {db_table_name} ({', '.join(cols)}) "
        f"SELECT DISTINCT {', '.join(['tmp.' + el for el in cols])} "
        f"FROM {db_table_name_tmp} AS tmp;"
    )
    add_rows = db.execute(text(sql_command)).rowcount
    db.execute(text(f"DROP TABLE {db_table_name_tmp} CASCADE;"))
    db.commit()
    return (
        f"добавлено {add_rows} записей"
        if update_rows == 0
        else f"добавлено {add_rows} записей, изменено {update_rows} записей"
    )
    # {"add_row": add_rows, "update_row": update_rows}


def find_headers_row(
    content,
    headers_list: list[str],
    sheet_name: str = "",
    number_analyzed_rows: int = 20,
    is_return_df: bool = False,
    skip_footer: int = 0,
):
    header_row = 0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
        if sheet_name != "":
            report_df = pd.read_excel(content, engine="openpyxl", sheet_name=sheet_name, skipfooter=skip_footer)
        else:
            report_df = pd.read_excel(content, engine="openpyxl", skipfooter=skip_footer)
    headers_present = [el for el in headers_list if el in list(report_df.columns)]
    if len(headers_list) != len(headers_present):
        max_of_matches = len(headers_present)
        for num_str in range(min(report_df.shape[0], number_analyzed_rows)):
            number_of_matches = sum(1 for el in headers_list if el in report_df.loc[num_str].values)
            if len(headers_list) == number_of_matches:
                header_row = num_str + 1
                break
            elif max_of_matches < number_of_matches:
                max_of_matches = number_of_matches
                headers_present = [el for el in headers_list if el in report_df.loc[num_str].values]
        if header_row == 0:
            raise HTTPException(
                status_code=422,
                detail=f"Некорректный формат файла: не найден следующий перечень колонок: "
                f"({', '.join(set(headers_list) - set(headers_present))})"
                # detail=f"The full list of headers was not found in the file: ({headers_list})"
            )
        if is_return_df:
            headers = report_df.loc[header_row - 1].values
            report_df = report_df[header_row:]
            report_df.columns = headers
    return report_df[headers_list].reset_index(drop=True) if is_return_df else header_row


def read_excel_with_find_headers_old(
    content, headers_list: list[str], sheet_name: str = "", number_analyzed_rows: int = 20, skip_footer: int = 0
):
    return find_headers_row(
        content=content,
        sheet_name=sheet_name,
        headers_list=headers_list,
        number_analyzed_rows=number_analyzed_rows,
        is_return_df=True,
        skip_footer=skip_footer,
    )


def read_excel_with_find_headers(
    file,
    headers_list: list[str],
    number_analyzed_rows: int = 20,
    converters: dict = {},
    sheet_name: str = "",
    skip_footer: int = 0,
    is_get_all_columns: bool = False,
    is_cumulative_headlines: bool = False,
    headers_list_min: list[str] = [],
):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
        if sheet_name != "":
            report_df = pd.read_excel(
                file,
                engine="openpyxl",
                sheet_name=sheet_name,
                converters=converters,
                skipfooter=skip_footer,
                index_col=False,
            )
        else:
            report_df = pd.read_excel(
                file, engine="openpyxl", converters=converters, skipfooter=skip_footer, index_col=False
            )
    headers_present_max = [el for el in headers_list if el in remove_br(report_df.columns)]
    headers_for_min = (
        [0, len(headers_present_max), remove_br(report_df.columns)]
        if headers_list_min and set(headers_list_min).issubset(set(headers_present_max))
        else [0, 0, []]
    )
    if len(headers_list) != len(headers_present_max):
        header_row = 0
        headers_in_row = [""] * len(report_df.columns)
        for num_str in range(min(report_df.shape[0], number_analyzed_rows)):
            if is_cumulative_headlines:
                headers_in_row = remove_br(
                    list(
                        map(
                            lambda x: str(x[0]) + " " + str(x[1]),
                            zip(headers_in_row, report_df.loc[num_str].fillna("").values),
                        )
                    )
                )
            else:
                headers_in_row = remove_br(report_df.loc[num_str].values)
            headers_present_cur = [el for el in headers_list if el in headers_in_row]
            if len(headers_list) == len(headers_present_cur):
                headers_present_max = headers_present_cur
                header_row = num_str + 1
                break
            elif len(headers_present_max) < len(headers_present_cur):
                headers_present_max = headers_present_cur
                headers_for_min = (
                    [num_str, len(headers_present_max), headers_present_max]
                    if headers_list_min and set(headers_list_min).issubset(set(headers_present_max))
                    else headers_for_min
                )

        # not all columns, but enough for a minimum
        if header_row == 0 and headers_for_min[1] > 0:
            header_row = headers_for_min[0]
            headers_in_row = headers_for_min[2]
            headers_present_max = [el for el in headers_list if el in headers_in_row]

        if header_row == 0 and headers_for_min[1] == 0:
            raise HTTPException(
                status_code=422,
                detail=f"Некорректный формат файла: не найден следующий перечень колонок: "
                f"({', '.join(set(headers_list) - set(headers_present_max))})",
            )

        # remove duplicates in column names
        for i in range(len(headers_in_row)):
            for j in range(i + 1, len(headers_in_row)):
                if headers_in_row[i] == headers_in_row[j]:
                    headers_in_row[j] += "_"

        # headers = report_df.loc[header_row - 1].values
        report_df = report_df[header_row:]
        report_df.columns = headers_in_row
        # report_df.columns = headers

    report_df.columns = remove_br(report_df.columns)
    return (
        report_df.reset_index(drop=True)
        if is_get_all_columns
        else report_df[headers_present_max].reset_index(drop=True)
    )


def remove_br(input_lst: list) -> list:
    """replace \n to space and remove double-space (often necessary for column headers in Excel)"""
    return [" ".join(str(el).split()) for el in input_lst]


def get_info_from_excel(content):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")  # for remove warn: "Workbook contains no default style, ..."
        try:
            report_df = pd.read_excel(content, engine="openpyxl")
        except Exception as e:
            try:
                report_df = pd.read_csv(content)
            except Exception as e:
                return {"rows": 0, "cols": 0}
    return {"rows": report_df.shape[0], "cols": report_df.shape[1]}


def read_sql_with_chunk(postgre_eng: Engine, sql_command: str, chunk_size: int = 100000) -> DataFrame:
    sys.stdout.write(f'Performing SQL: "{sql_command}"')
    offset = 0
    report_df = DataFrame()
    while True:
        sql = sql_command + f" limit {chunk_size} offset {offset}"
        df = pd.read_sql(sql, con=postgre_eng)
        # df = reduce_mem_usage(df)
        report_df = pd.concat([report_df, df])
        offset += chunk_size
        sys.stdout.write(".")
        sys.stdout.flush()
        if df.shape[0] < chunk_size:
            break
    report_df.reset_index(drop=True, inplace=True)
    return report_df


def optimize_memory_usage(df, print_size=True):
    # Function optimizes memory usage in dataframe.
    # (RU) Функция оптимизации типов в dataframe.

    # Types for optimization.
    # Типы, которые будем проверять на оптимизацию.
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    # Memory usage size before optimize (Mb).
    # (RU) Размер занимаемой памяти до оптимизации (в Мб).
    before_size = df.memory_usage().sum() / 1024**2
    for column in df.columns:
        column_type = df[column].dtypes
        if column_type in numerics:
            column_min = df[column].min()
            column_max = df[column].max()
            if str(column_type).startswith("int"):
                if column_min > np.iinfo(np.int8).min and column_max < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
                elif column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                elif column_min > np.iinfo(np.int64).min and column_max < np.iinfo(np.int64).max:
                    df[column] = df[column].astype(np.int64)
            else:
                if column_min > np.finfo(np.float32).min and column_max < np.finfo(np.float32).max:
                    df[column] = df[column].astype(np.float32)
                else:
                    df[column] = df[column].astype(np.float64)
                    # Memory usage size after optimize (Mb).
    # (RU) Размер занимаемой памяти после оптимизации (в Мб).
    after_size = df.memory_usage().sum() / 1024**2
    if print_size:
        print(
            "Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).".format(
                before_size, after_size, 100 * (before_size - after_size) / before_size
            )
        )
    return df


def is_number(may_be_number):
    # equal str(may_be_number).replace('.', '', 1).isdigit()
    try:
        float(may_be_number)
        return True
    except ValueError:
        return False


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def transliteration(text):
    cyrillic = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    latin = "a|b|v|g|d|e|e|zh|z|i|i|k|l|m|n|o|p|r|s|t|u|f|kh|tc|ch|sh|shch||y||e|iu|ia".split("|")
    tran_dict = {k: v for k, v in zip(cyrillic, latin)}
    new_text = ""
    for letter in text:
        new_letter = tran_dict.get(letter.lower(), letter)
        new_text += new_letter if letter.islower() else new_letter.upper()
    return new_text


def remove_markup_from_str(text: str) -> str:
    # remove the markup from the content
    return re.sub(r"|".join(map(re.escape, ["<b>", "</b>", "<u>", "</u>", "<i>", "</i>", "<pre>", "</pre>"])), "", text)


def timedelta_to_dhms(duration):
    # преобразование в дни, часы, минуты и секунды
    class TimeDelta:
        __slots__ = ["days", "hours", "minutes", "seconds"]

    result = TimeDelta()
    result.days = duration.days
    result.hours = f"{(duration.seconds // 3600):02}"
    result.minutes = f"{((duration.seconds % 3600) // 60):02}"
    result.seconds = f"{((duration.seconds % 60)):02}"
    return result


def is_date(date_str: str) -> bool:
    try:
        return bool(date_parser.parse(date_str))
    except ValueError:
        return False


def is_date_matching(date_str):
    try:
        return bool(datetime.strptime(date_str, "%Y-%m-%d"))
    except ValueError:
        return False


def date_list_to_str(date_list: list):
    try:
        if len(date_list) == 1:
            date_list_str = f"{date_list[0]}"
        elif (max(date_list) - min(date_list)).days + 1 == len(date_list):
            date_list_str = f"{date_list[0]} - {date_list[-1]}"
        else:
            date_list_str = f"{', '.join(map(lambda x: f'{x}', date_list))}"
    except:
        date_list_str = ""
    return date_list_str


def is_excel_by_content_type(content_type: str) -> bool:
    """
    Extension 	MIME Type (Type / SubType)	Kind of Document
    .xls	application/vnd.ms-excel	 Microsoft Excel
    .xlsx	application/vnd.openxmlformats-officedocument.spreadsheetml.sheet	 Microsoft Excel (OpenXML)
    .xltx 	application/vnd.openxmlformats-officedocument.spreadsheetml.template	Office Excel 2007 template
    .xlsm	application/vnd.ms-excel.sheet.macroEnabled.12	 Office Excel 2007 macro-enabled workbook
    .xltm	application/vnd.ms-excel.template.macroEnabled.12	Office Excel 2007 macro-enabled workbook template
    .xlam	application/vnd.ms-excel.addin.macroEnabled.12	Office Excel 2007 macro-enabled add-in
    .xlsb	application/vnd.ms-excel.sheet.binary.macroEnabled.12	Office Excel 2007 non xml binary workbook
    """
    return True if ("excel" in content_type or "openxmlformats-officedocument" in content_type) else False


def df_from_sql(query):
    return DataFrame([el.__dict__ for el in query]).drop(columns="_sa_instance_state") if len(query) else DataFrame()


def dict_from_db_old(model):
    return {el: model.__dict__[el] for el in model.__dict__ if
            not isinstance(model.__dict__[el], InstanceState | InstrumentedList)}


def dict_from_db(model, instance_list: list = [InstanceState]):
    if type(model) == InstrumentedList:
        return [dict_from_db(el, instance_list) for el in model]
    result = {el: model.__dict__[el] for el in model.__dict__ if
            not isinstance(model.__dict__[el], tuple(instance_list))}
    # against looping during recursion
    instance_list = instance_list + [type(model)]
    for key, val in result.items():
        if type(val) == InstrumentedList:
            result[key] = dict_from_db(val, instance_list)
    return result


def list_of_dict_from_sql_row(input_list: list) -> list:
    if isinstance(input_list, list):
        try:
            return [{key: val for key, val in zip(row._fields, row._data)} for row in input_list]
        except:
            return input_list
    return []


def model_columns(model: DeclarativeMeta) -> dict:
    return {el.key: el.comment for el in model.__table__.columns}


def model_to_df_empty(model: DeclarativeMeta) -> list:
    return DataFrame(columns=[el.key for el in model.__table__.columns])


def df_replace_true_false_with_yes_no(df: DataFrame, model_class: BaseModel) -> DataFrame:
    # convert "True/False" to "да/нет"
    if df.shape[0] > 0:
        cols_db = {el.key: el.type for el in model_class.__table__.columns}
        for col in df.columns:
            try:
                if is_bool_dtype(df[col]) or cols_db[col].python_type == bool:
                    df.fillna({col: ""}, inplace=True)
                    df.loc[df[col] == True, col] = "да"
                    df.loc[df[col] == False, col] = "нет"
                    df[col] = df[col].astype(str)
            except:
                pass
    return df


def df_convert_number_to_number(df: DataFrame) -> DataFrame:
    # replace dot with comma for Decimal
    if df.shape[0] > 0:
        df = df.copy()
        # non-date format columns
        for column in [column for column in df.columns if not is_datetime(df[column])]:
            # df[column] = df[column].apply(pd.to_numeric, errors="ignore")
            try:
                df[column] = pd.to_numeric(df[column], errors="raise")
            except (ValueError, TypeError):
                # Same logic as errors='ignore' in pd.to_numeric
                pass

    # return df.apply(pd.to_numeric, errors="ignore")
    return df


def df_column_to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def df_column_excel_int_to_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="ignore")
    except:
        pass
    try:
        return pd.to_datetime(series, errors="ignore")
    except:
        return series


def max_digit_after_point(series: Series) -> int:
    return max(series.astype(str).str.extract(r'\.(.*)', expand=False).str.len()) - 1


class DashboardDate(BaseModel):
    date_from: str = str(datetime.today().replace(day=1, month=1).date())
    date_to: str = str(datetime.today().date())

    @root_validator(pre=True)
    def date_validation(cls, values):
        try:
            values["date_from"] = str(parse(values["date_from"]))
            values["date_to"] = f"{parse(values['date_to']).date()} 23:59:59"
            # values['date_to'] = str(parse(values['date_to']).date())
            # values['date_from'] = datetime.strptime(values['date_from'], '%Y-%m-%d').isoformat()
            # values['date_to'] = datetime.strptime(values['date_to'], '%Y-%m-%d').isoformat()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        if values["date_from"] > values["date_to"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Начальная дата должна быть меньше или равна конечной дате.",
            )
        return values


def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_quarter(p_date: date_class) -> int:
    return (p_date.month - 1) // 3 + 1


def get_first_day_of_the_quarter(p_date: date_class):
    return datetime(p_date.year, 3 * ((p_date.month - 1) // 3) + 1, 1)


def get_last_day_of_the_quarter(p_date: date_class):
    quarter = get_quarter(p_date)
    return datetime(p_date.year + 3 * quarter // 12, 3 * quarter % 12 + 1, 1) + timedelta(days=-1)


def get_first_day_of_the_year(p_date: date_class):
    return datetime(p_date.year, 1, 1)


def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator
