import datetime
import types
from typing import Optional, Mapping, Any

from pydantic import BaseModel, BaseConfig
from pydantic.fields import ModelField, FieldInfo
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeMeta


class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True


class SimpleScheme(OurBaseModel):
    id: int
    name: str


def convert_to_optional(schema):
    return {k: Optional[v] for k, v in schema.__annotations__.items()}


def copy_class(c, name=None):
    if not name:
        name = "CopyOf" + c.__name__
    if hasattr(c, "__slots__"):
        slots = c.__slots__ if type(c.__slots__) != str else (c.__slots__,)
        dict_ = dict()
        sloted_members = dict()
        for k, v in c.__dict__.items():
            if k not in slots:
                dict_[k] = v
            elif type(v) != types.MemberDescriptorType:
                sloted_members[k] = v
        CopyOfc = type(name, c.__bases__, dict_)
        for k, v in sloted_members.items():
            setattr(CopyOfc, k, v)
        return CopyOfc
    else:
        dict_ = dict(c.__dict__)
        return type(name, c.__bases__, dict_)


def model_annotations_with_parents(model: BaseModel) -> Mapping[str, Any]:
    parent_models: list[type] = [
        parent_model
        for parent_model in model.__bases__
        if (issubclass(parent_model, BaseModel) and hasattr(parent_model, "__annotations__"))
    ]
    annotations: Mapping[str, Any] = {}
    for parent_model in reversed(parent_models):
        annotations.update(model_annotations_with_parents(parent_model))

    annotations.update(model.__annotations__)
    return annotations


def copy_model_factory(model: BaseModel, name: str = None) -> BaseModel:
    if not name:
        name = f"{model.__name__}_copy"
    return type(
        name,
        (model,),
        dict(
            __module__=model.__module__,
            __annotations__={k: v for k, v in model_annotations_with_parents(model).items()},
        ),
    )


def partial_model_factory(model: BaseModel, prefix: str = "Partial", name: str = None) -> BaseModel:
    if not name:
        name = f"{prefix}{model.__name__}"
    return type(
        name,
        (model,),
        dict(
            __module__=model.__module__,
            __annotations__={str(k): Optional[v] for k, v in model_annotations_with_parents(model).items()},
        ),
    )


def partial_model(cls: BaseModel) -> BaseModel:
    return partial_model_factory(cls, name=cls.__name__ + "_partition")


class SchemaGenerate:
    """
    A universal class for dynamically creating SchemaCreate, SchemaUpdate and Schema based on the Model description.
    Schema - the schema includes all fields of the model with types, annotations, and required attributes
    SchemaCreate - identical to Schema but does not contain keys created automatically
    SchemaUpdate - identical to SchemaCreate, but all fields are optional (the so-called PARTIAL_SCHEMA)
    """

    schema_create: BaseModel = None
    schema_update: BaseModel = None
    schema: BaseModel = None

    _type_mapping = {
        "INTEGER": int,
        "BIGINT": int,
        "BigInteger": int,
        "SmallInteger": int,
        "Float": float,
        "NUMERIC": float,
        "Date": datetime.date,
        "DateTime": datetime.datetime,
        "Time": datetime.time,
        "Interval": datetime.timedelta,
        "String": str,
        "VARCHAR": str,
        "Text": str,
        "Unicode": str,
        "UnicodeText": str,
        "Boolean": bool,
        "LargeBinary": bytes,
        "BYTEA": bytes,
        "Enum": str,
        "MatchType": str,
    }

    class OurBaseModel(BaseModel):
        class Config:
            orm_mode = True

    def __init__(
        self,
        model: DeclarativeMeta,
        schema: BaseModel = None,
        schema_create: BaseModel = None,
        schema_update: BaseModel = None,
    ):
        self._model = model
        self._columns_key = self._get_columns_key()
        self._columns_required = self._get_columns_required()
        self.schema_create = schema_create if schema_create else self._generate_schema(is_column_key_include=False)
        self.schema_update = schema_update if schema_update else self._generate_schema_update()
        self.schema = schema if schema else self._generate_schema(is_column_key_include=True)

    def _generate_schema(self, is_column_key_include: bool = False):
        schema_name = f"Schema{self._model.__name__}" + ("" if is_column_key_include else "Create")
        result = copy_model_factory(self.OurBaseModel, schema_name)
        for el in self._model.__table__.columns:
            # without 'id' with autoincrement
            if is_column_key_include or el.key not in self._columns_key:
                type_ = self._get_column_type(el)
                required = True if el.key in self._columns_required else False
                field = ModelField(
                    name=str(el.key),
                    type_=type_,
                    class_validators={},
                    default=None,
                    required=required,
                    model_config=BaseConfig,
                    field_info=FieldInfo(None),
                )
                result.__fields__[el.key] = field
                annotation = self._get_column_annotation(el, required)
                if annotation:
                    result.__annotations__[el.key] = annotation
        return result

    def _generate_schema_update(self):
        return partial_model(self.schema_create)

    def _get_columns_key(self) -> list:
        return [el.key for el in self._model.__table__.columns if el.primary_key and el.autoincrement]

    def _get_columns_unique(self) -> list:
        result = []
        for el in self._model.__table_args__:
            if isinstance(el, UniqueConstraint):
                result.append([col.key for col in el.columns._all_columns])
        columns_unique = [el.key for el in self._model.__table__.columns if el.unique]
        result.extend([[el] for el in columns_unique])
        return result

    def _get_columns_required(self) -> list:
        result = []
        if "__table_args__" in self._model.__dict__:
            for el in self._model.__table_args__:
                if isinstance(el, UniqueConstraint):
                    result.extend([col.key for col in el.columns._all_columns])
        columns_required = [
            el.key for el in self._model.__table__.columns if not el.nullable or el.unique or el.foreign_keys
        ]
        result.extend([el for el in columns_required])
        # result = list(set(result) - set(self._columns_key))
        return list(set(result))

    def _get_column_type(self, column):
        type_ = str(column.type).upper()
        result = [val for key, val in self._type_mapping.items() if type_.startswith(key.upper())]
        if not result:
            print(f'The data type {type_} cannot be processed - we consider it as "str"')
        return result[0] if result else str

    def _get_column_annotation(self, column, required: bool):
        type_ = str(column.type).upper()
        result = [val for key, val in self._type_mapping.items() if type_.startswith(key.upper())]
        return (result[0] if required else (result[0] | None)) if result else None
