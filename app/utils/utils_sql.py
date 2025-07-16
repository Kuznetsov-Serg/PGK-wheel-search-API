from typing import Union

from fastapi import HTTPException
from sqlalchemy.orm import DeclarativeMeta

from app.utils.utils import is_number, is_date, camel_to_snake


class BuildSQL:
    _type_mapping = {
        "INTEGER": "number",
        "BIGINT": "number",
        "BigInteger": "number",
        "SmallInteger": "number",
        "Float": "number",
        "NUMERIC": "number",
        "Date": "date",
        "DateTime": "date",
        "Time": "text",
        "Interval": "text",
        "String": "text",
        "VARCHAR": "text",
        "Text": "text",
        "Unicode": "text",
        "UnicodeText": "text",
        "Boolean": "text",
        "LargeBinary": "text",
        "BYTEA": "text",
        "Enum": "text",
        "MatchType": "text",
    }

    @classmethod
    def get_sql_condition(
        cls,
        filter_model: dict = dict(),
        sort_model: Union[dict, list] = dict(),
        sort_default: str = None,
        model: DeclarativeMeta = None,
    ):
        """
        filter_model = {
                # simple к примеру: {'branch_name': 'московск', 'date_detect': '2022-10-23', 'num': 10}
                'field_name': string | number | date
                # or complex, к примеру: {"branch_name": {"filterType": "text", "type": "contains", "filter": "но"}}
                'field_name': {
                                'filter' | 'filterFrom' & 'filterTo'    # операнды
                                'type': ...     # тип оператора( = правило сравнения), equals, lessThan...
                                'filterType': date | text | number  # тип данных (optional: if None - get from model)
                                },
            }
            type in ['contains', 'notContains', 'equals', 'notEqual', 'startsWith', 'endsWith', 'blank', 'notBlank',
                        'bool', 'greaterThan', 'greaterThanOrEqual', 'lessThan', 'lessThanOrEqual', 'inRange', 'inList']
        -----------------------------------------------------------------------------------------------------------
        "sort_model" = {"col_id": "col_name", "sort": "asc" | "desc"}
        or
        "sort_model"= [{"col_id": "col_name_1", "sort": "asc" | "desc"}, ...,
                       {"colId": "col_name_N", "sort": "asc" | "desc"}]
        """
        # add filtering by any fields
        filter_command = (
            " and ".join([f"({cls._parse_filter(field, value, model)})" for field, value in filter_model.items()])
            if filter_model
            else ""
        )
        # add sorting by any field
        if sort_model and len(sort_model):
            sort_model = [sort_model] if type(sort_model) is dict else sort_model
        else:
            sort_model = []
        # remove empty dict
        sort_model = [el for el in sort_model if (("col_id" in el) or ("colId" in el))]
        sort_command = (
            f"ORDER BY "
            + ", ".join(
                [
                    f"{el['col_id'] if 'col_id' in el else el['colId']} " f"{el['sort'] if 'sort' in el else 'asc'}"
                    for el in sort_model
                ]
            )
            if sort_model and len(sort_model)
            else f" order by {sort_default}"
            if sort_default
            else ""
        )
        return f"{filter_command} {sort_command}"

    @staticmethod
    def _check_parameters(field: str, filter_obj: dict):
        begin_exception = f"For field='{field}' parameter "
        if "type" not in filter_obj:
            raise HTTPException(status_code=404, detail=begin_exception + f"'type' not found")
        if filter_obj["type"] != "bool":
            if "filterType" not in filter_obj:
                raise HTTPException(status_code=404, detail=begin_exception + f"'filterType' not found")
            if filter_obj["filterType"] not in ["number", "text", "date"]:
                raise HTTPException(
                    status_code=404, detail=begin_exception + f"'filterType' can only be 'number', 'text' or 'date'"
                )
        if filter_obj["type"] not in ["blank", "notBlank"]:
            if filter_obj["type"] in ("inRange",):
                if ("filterFrom" not in filter_obj) or ("filterTo" not in filter_obj):
                    raise HTTPException(
                        status_code=404, detail=begin_exception + f"'filterFrom' or 'filterTo' not found"
                    )
            elif "filter" not in filter_obj:
                raise HTTPException(status_code=404, detail=begin_exception + f"'filter' not found")
            elif filter_obj["type"] == "bool" and filter_obj["filter"].lower() not in ["true", "false", "null"]:
                raise HTTPException(
                    status_code=404, detail=begin_exception + f"'type' can only be 'TRUE', 'FALSE' or 'NULL'"
                )
            if filter_obj["type"] in ("inList",) and type(filter_obj["filter"]) is not list:
                raise HTTPException(
                    status_code=404, detail=begin_exception + f"'type'=='inlist' ==> 'filter' mast be List."
                )

    @classmethod
    def _parse_filter(cls, field, filter_obj, model) -> str:
        # get the field type from the model, if it is specified
        if model:
            postgres_sql_type = str(model.__table__.columns[field].type).upper()
            result = [val for key, val in cls._type_mapping.items() if postgres_sql_type.startswith(key.upper())]
        else:
            result = None
        field_type = result[0] if result else "text"

        # simple filter_model (for example: {'branch_name': 'московск'})
        if not type(filter_obj) is dict:
            if not field_type:
                field_type = "number" if is_number(filter_obj) else "date" if is_date(filter_obj) else "text"
            return (
                f"{field} = {filter_obj}"
                if field_type == "number"
                else f"{field} = '{filter_obj}'"
                if filter_obj == "date"
                else f"{field} ILIKE '%{filter_obj}%'"
            )

        if "filterType" not in filter_obj:
            filter_obj["filterType"] = field_type
        cls._check_parameters(field, filter_obj)
        try:
            filter_method = getattr(cls, f"_{camel_to_snake(filter_obj['type'])}")
        except Exception:
            raise HTTPException(400, f"The system does not support 'type' = '{filter_obj['type']}' (field='{field}').")
        return filter_method(field, filter_obj)

    @staticmethod
    def _equals(field, filter_obj) -> str:
        if filter_obj["filterType"] == "number":
            result = f"= {filter_obj['filter']}"
        elif filter_obj["filterType"] == "date":
            result = f"= '{filter_obj['filter']}'"
        else:
            result = f"ILIKE '{filter_obj['filter']}'"
        return f"{field} {result}"

    @staticmethod
    def _not_equal(field, filter_obj) -> str:
        if filter_obj["filterType"] == "number":
            result = f"!= {filter_obj['filter']}"
        elif filter_obj["filterType"] == "date":
            result = f"!= '{filter_obj['filter']}'"
        else:
            result = f"NOT ILIKE '{filter_obj['filter']}'"
        return f"{field} {result}"

    @staticmethod
    def _greater_than(field, filter_obj) -> str:
        return (
            f"{field} > {filter_obj['filter']}"
            if filter_obj["filterType"] == "number"
            else f"{field} > '{filter_obj['filter']}'"
        )

    @staticmethod
    def _greater_than_or_equal(field, filter_obj) -> str:
        return (
            f"{field} >= {filter_obj['filter']}"
            if filter_obj["filterType"] == "number"
            else f">= '{filter_obj['filter']}'"
        )

    @staticmethod
    def _less_than(field, filter_obj) -> str:
        return (
            f"{field} < {filter_obj['filter']}"
            if filter_obj["filterType"] == "number"
            else f"{field} < '{filter_obj['filter']}'"
        )

    @staticmethod
    def _less_than_or_equal(field, filter_obj) -> str:
        return (
            f"{field} <= {filter_obj['filter']}"
            if filter_obj["filterType"] == "number"
            else f"{field} <= '{filter_obj['filter']}'"
        )

    @staticmethod
    def _contains(field, filter_obj) -> str:
        return (
            f"{field} ILIKE '%{filter_obj['filter']}%'"
            if filter_obj["filterType"] == "text"
            else f"CAST({field} as text) ILIKE '%{filter_obj['filter']}%'"
        )

    @staticmethod
    def _not_contains(field, filter_obj) -> str:
        return (
            f"{field} NOT ILIKE '%{filter_obj['filter']}%'"
            if filter_obj["filterType"] == "text"
            else f"CAST({field} as text) NOT ILIKE '%{filter_obj['filter']}%'"
        )

    @staticmethod
    def _starts_with(field, filter_obj) -> str:
        return (
            f"{field} ILIKE '{filter_obj['filter']}%'"
            if filter_obj["filterType"] == "text"
            else f"CAST({field} as text) ILIKE '{filter_obj['filter']}%'"
        )

    @staticmethod
    def _ends_with(field, filter_obj) -> str:
        return (
            f"{field} ILIKE '%{filter_obj['filter']}'"
            if filter_obj["filterType"] == "text"
            else f"CAST({field} as text) ILIKE '%{filter_obj['filter']}'"
        )

    @staticmethod
    def _in_range(field, filter_obj) -> str:
        return (
            f"{field} between {filter_obj['filterFrom']} and {filter_obj['filterTo']}"
            if filter_obj["filterType"] == "number"
            else f"{field} between '{filter_obj['filterFrom']}' and '{filter_obj['filterTo']}'"
        )

    @staticmethod
    def _in_list(field, filter_obj) -> str:
        return (
            f"{field} IS NULL"  # for an empty list of values
            if len(filter_obj["filter"]) == 0
            else f"{field} IN (" + ", ".join(map(lambda x: f"{x}", filter_obj["filter"])) + ")"  # for number
            if filter_obj["filterType"] == "number"
            else f"{field} IN (" + ", ".join(map(lambda x: f"'{x}'", filter_obj["filter"])) + ")"  # for text/date
        )

    @staticmethod
    def _blank(field, filter_obj) -> str:
        result = f"{field} IS NULL"  # enough for 'filterType' == 'date'
        if filter_obj["filterType"] == "number":
            result += f" OR {field} = 0"
        elif filter_obj["filterType"] == "text":
            result += f" OR {field} = ''"
        return result

    @staticmethod
    def _not_blank(field, filter_obj) -> str:
        result = f"{field} IS NOT NULL"  # enough for 'filterType' == 'date'
        if filter_obj["filterType"] == "number":
            result += f" AND {field} <> 0"
        elif filter_obj["filterType"] == "text":
            result += f" AND {field} <> ''"
        return result

    @staticmethod
    def _bool(field, filter_obj) -> str:
        # return f"{field} IS TRUE" if filter_obj["filter"].lower == "true" else f"{field} IS NOT TRUE"  # FALSE = NULL
        return f"{field} IS {filter_obj['filter']}"  # ignores NULL
