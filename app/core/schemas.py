import datetime
from typing import Any, Mapping, Optional

from pydantic import BaseModel


# GENERAL SERVICE SCHEMES


class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True


class ResponseAnswer(BaseModel):
    message: str
    log_id: int


def convert_to_optional(schema):
    return {k: Optional[v] for k, v in schema.__annotations__.items()}


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


def partial_model_factory(model: BaseModel, prefix: str = "Partial", name: str = None) -> BaseModel:
    if not name:
        name = f"{prefix}{model.__name__}"

    return type(
        name,
        (model,),
        dict(
            __module__=model.__module__,
            __annotations__={k: Optional[v] for k, v in model_annotations_with_parents(model).items()},
        ),
    )


def partial_model(cls: BaseModel) -> BaseModel:
    return partial_model_factory(cls, name=cls.__name__)


# USER DEFINED SCHEMAS


# PROJECT "DAMAGE COMPENSATION"
class WorkFilterBase(OurBaseModel):
    name: str


class WorkFilterCreate(WorkFilterBase):
    pass


@partial_model
class WorkFilterUpdate(WorkFilterCreate):
    pass


class WorkFilter(WorkFilterBase):
    id: int
    storage_list: list


class WorkFilterStorageBase(OurBaseModel):
    work_filter_id: int
    branch_name: str
    storage_name: str
    number_rdv: (str | None)
    work_name_by_document: (str | None)


class WorkFilterStorageCreate(WorkFilterStorageBase):
    pass


@partial_model
class WorkFilterStorageUpdate(WorkFilterStorageCreate):
    pass
    # __annotations__ = convert_to_optional(WorkFilterStorageCreate)


class WorkFilterStorage(WorkFilterStorageBase):
    id: int


class WorkCostBase(OurBaseModel):
    name: str
    number: str
    cost: float


class WorkCostCreate(WorkCostBase):
    pass


@partial_model
class WorkCostUpdate(WorkCostCreate):
    pass


class WorkCost(WorkCostBase):
    id: int


class StealBase(OurBaseModel):
    date_upload: (datetime.date | None) = None
    document_num: (int | None)
    document_num_num: (int | None)
    period_load: (str | None)
    date_detect: (datetime.date | None) = None
    branch_name: (str | None)
    storage_name: (str | None)
    wagon_num: int
    external_contract_num: (str | None)
    part_name: str
    part_cost: float
    part_amount: int
    repay: (float | None)
    repay_author: (float | None)
    insurance_notification_date: (datetime.date | None) = None
    insurance_notification_num: (str | None)
    insurance_name: (str | None)
    is_insurance_of_carrier: (bool | None) = True
    insurance_number: (str | None)
    insurance_claim_number: (str | None)
    insurance_payment_total: (float | None)
    insurance_payment_date: (datetime.date | None) = None
    insurance_payment_done: (float | None)
    author_pretension_date: (datetime.date | None) = None
    author_name: (str | None)
    author_pretension_number: (str | None)
    author_payment_total: (float | None)
    author_payment_date: (datetime.date | None) = None
    author_payment_done: (float | None)
    author_lawyer_date: (datetime.date | None) = None
    author_lawyer_number: (str | None)
    police_date: (datetime.date | None) = None
    police_ovd_date: (datetime.date | None) = None
    police_ovd_name: (str | None)
    police_payment: (float | None)
    police_decision: (str | None)
    police_decision_date: (datetime.date | None) = None


class StealCreate(StealBase):
    document_num: int
    document_num_num: int


@partial_model
class StealUpdate(StealCreate):
    pass


class Steal(StealBase):
    id: int


class CheckFilterBase(OurBaseModel):
    filter_name: str
    work_name: str
    distance: int
    count: int


class CheckFilterCreate(WorkCostBase):
    pass


@partial_model
class CheckFilterUpdate(WorkCostCreate):
    pass


class CheckFilter(WorkCostBase):
    id: int


steal_config_CRUD = {
    "order_by": "date_detect",
    "sort_model": [{"col_id": "date_detect", "sort": "desc"}, {"col_id": "branch_name"}, {"col_id": "storage_name"}],
}
work_filter_config_CRUD = {"filter_by": "name", "sort_model": [{"col_id": "name"}]}
work_filter_storage_config_CRUD = {
    "filter_by": "branch_name",
    "sort_model": [{"col_id": "branch_name"}, {"col_id": "storage_name"}],
}
work_cost_config_CRUD = {"filter_by": "name", "sort_model": [{"col_id": "name"}, {"col_id": "number"}]}
check_filter_config_CRUD = {"filter_by": "filter_name", "sort_model": [{"col_id": "filter_name"}]}


# PROJECT "WHEEL SEARCH"
class WheelSetBase(OurBaseModel):
    date_upload: Optional[datetime.date] = None
    load_from: (str | None)
    wagon_num: (int | None)
    branch_name: (str | None)
    railway: (str | None)
    date_detect: Optional[datetime.date] = None
    part_number: str
    thickness_left_rim: (int | None)
    thickness_right_rim: (int | None)
    rejection_code: (str | None)
    part_cost: (float | None)
    mounting_type: (str | None)
    system: (str | None)
    storage_name: (str | None)
    external_contract_num: (str | None)
    document_num: (int | None)
    repay: (float | None)
    repay_author: (float | None)
    insurance_notification_date: (datetime.date | None) = None
    insurance_notification_num: (str | None)
    insurance_name: (str | None)
    is_insurance_of_carrier: (bool | None) = True
    insurance_number: (str | None)
    insurance_claim_number: (str | None)
    insurance_payment_total: (float | None)
    insurance_payment_date: (datetime.date | None) = None
    insurance_payment_done: (float | None)
    author_pretension_date: (datetime.date | None) = None
    author_name: (str | None)
    author_pretension_number: (str | None)
    author_payment_total: (float | None)
    author_payment_date: (datetime.date | None) = None
    author_payment_done: (float | None)
    author_lawyer_date: (datetime.date | None) = None
    author_lawyer_number: (str | None)
    police_date: (datetime.date | None) = None
    police_ovd_date: (datetime.date | None) = None
    police_ovd_name: (str | None)
    police_payment: (float | None)
    police_decision: (str | None)
    police_decision_date: (datetime.date | None) = None
    # 1
    claim_refund_partner_1: (str | None)
    claim_date_1: (datetime.date | None) = None
    claim_internal_number_1: (str | None)
    claim_external_number_1: (str | None)
    claim_refund_date_1: (datetime.date | None) = None
    claim_sum_1: (float | None)
    claim_sum_damage_1: (float | None)
    claim_partner_payed_1: (float | None)
    # 2
    claim_refund_partner_2: (str | None)
    claim_date_2: (datetime.date | None) = None
    claim_internal_number_2: (str | None)
    claim_external_number_2: (str | None)
    claim_refund_date_2: (datetime.date | None) = None
    claim_sum_2: (float | None)
    claim_sum_damage_2: (float | None)
    claim_partner_payed_2: (float | None)
    # 3
    claim_refund_partner_3: (str | None)
    claim_date_3: (datetime.date | None) = None
    claim_internal_number_3: (str | None)
    claim_external_number_3: (str | None)
    claim_refund_date_3: (datetime.date | None) = None
    claim_sum_3: (float | None)
    claim_sum_damage_3: (float | None)
    claim_partner_payed_3: (float | None)
    # sum
    claim_sum_all: (float | None)
    claim_sum_damage_all: (float | None)
    claim_partner_payed_all: (float | None)


class WheelSetCreate(WheelSetBase):
    date_detect: datetime.date
    part_number: str
    storage_name: str


@partial_model
class WheelSetUpdate(WheelSetCreate):
    pass


class WheelSet(WheelSetBase):
    id: int


class WheelSetCostBase(OurBaseModel):
    name: str
    rim_thickness_min: int
    rim_thickness_max: int
    mounting_type: str
    cost: float


class WheelSetCostCreate(WheelSetCostBase):
    pass


@partial_model
class WheelSetCostUpdate(WheelSetCostCreate):
    pass


class WheelSetCost(WheelSetCostBase):
    id: int


class WheelSetFilterBase(OurBaseModel):
    name: str


class WheelSetFilterCreate(WheelSetFilterBase):
    pass


@partial_model
class WheelSetFilterUpdate(WheelSetFilterCreate):
    pass


class WheelSetFilter(WheelSetFilterBase):
    id: int


class StorageBase(OurBaseModel):
    name: str
    branch: (str | None)
    railway: (str | None)


class StorageCreate(StorageBase):
    pass


@partial_model
class StorageUpdate(StorageCreate):
    pass


class Storage(StorageBase):
    id: int


class StorageBase(OurBaseModel):
    name: str
    branch: (str | None)
    railway: (str | None)


class StorageCreate(StorageBase):
    pass


@partial_model
class StorageUpdate(StorageCreate):
    pass


class Storage(StorageBase):
    id: int


wheel_set_config_CRUD = {
    "order_by": "date_detect",
    "sort_model": [{"col_id": "date_detect", "sort": "desc"}, {"col_id": "branch_name"}, {"col_id": "storage_name"}],
}
wheel_set_filter_config_CRUD = {"filter_by": "name", "sort_model": [{"col_id": "name"}]}
wheel_set_cost_config_CRUD = {
    "filter_by": "branch_name",
    "sort_model": [
        {"col_id": "mounting_type"},
        {"col_id": "is_same_cost_for_all_branch", "sort": "desc"},
        {"col_id": "branch_name"},
        {"col_id": "rim_thickness_min"},
    ],
}
mounting_type_map_config_CRUD = {
    "filter_by": "name_from",
    "sort_model": [{"col_id": "name_to"}, {"col_id": "name_from"}],
}
mounting_type_config_CRUD = {"filter_by": "name", "sort_model": [{"col_id": "name"}]}
storage_config_CRUD = {"filter_by": "name", "sort_model": [{"col_id": "name"}]}
