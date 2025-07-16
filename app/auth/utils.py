from itertools import zip_longest

from sqlalchemy.orm import Session

from app.auth.models import UserRole, RoleItem, TypeCRUD, TYPE_CRUD_COMMENT
from app.settings import PARSED_CONFIG


def get_user_access_dict(db: Session, user_id: int, is_only_in_role: bool = True, ad_restriction: dict = None) -> dict:
    result = dict() if is_only_in_role else get_all_access_false()
    role = db.query(UserRole).filter(UserRole.user_id == user_id).all()
    if role:
        role_item = db.query(RoleItem).filter(RoleItem.role_id.in_([el.role_id for el in role])).all()
        # result = {el.item: el.access for el in role_item}
        for el in role_item:
            if el.item in result:
                result[el.item] = "".join(
                    [(el1 if el1 == "1" else el2) for el1, el2 in zip_longest(result[el.item], el.access, fillvalue=0)]
                )
            else:
                result[el.item] = el.access
    if ad_restriction:
        result = restriction_minus(result, ad_restriction)
    return {key: access_convert_str_to_dict(val) for key, val in result.items()}


def get_all_access_false() -> dict:
    return {table: "0" * TypeCRUD.__len__() for table in PARSED_CONFIG.tables_tracked_for_authorization}


def get_all_access_true() -> dict:
    return {table: "1" * TypeCRUD.__len__() for table in PARSED_CONFIG.tables_tracked_for_authorization}


def get_all_access_false_bool() -> dict:
    return {key: access_convert_str_to_dict(val) for key, val in get_all_access_false().items()}


def get_all_access_true_bool() -> dict:
    return {key: access_convert_str_to_dict(val) for key, val in get_all_access_true().items()}


def get_role_access_dict(db: Session, role_id: int) -> dict:
    result = db.query(RoleItem).filter(RoleItem.role_id == role_id).all()
    return {el.item: access_convert_str_to_dict(el.access) for el in result} if result else dict()


def access_convert_str_to_dict(access: str) -> dict:
    return {el.value: (True if str(access + "0" * 20)[count] == "1" else False) for count, el in enumerate(TypeCRUD)}


def access_convert_dict_to_str(access: dict = dict()) -> str:
    result = "".join(["1" if (el.value in access and access[el.value]) else "0" for el in TypeCRUD])
    return result


def get_crud_type() -> dict:
    return {el.value: TYPE_CRUD_COMMENT[el] for el in TypeCRUD}


def restriction_sum(restriction_1: dict, restriction_2: dict) -> dict:
    def restriction_add(restriction: dict):
        for key, val in restriction.items():
            if key in result:
                result[key] = "".join(
                    [
                        (el1 if el1 == "1" else el2)
                        for el1, el2 in zip_longest(result[key], val, fillvalue=0)
                    ]
                )
            else:
                result[key] = val

    result = dict()
    restriction_add(restriction_1)
    restriction_add(restriction_2)
    return result


def restriction_minus(restriction_from: dict, restriction: dict) -> dict:
    result = restriction_from.copy()
    for key, val in restriction.items():
        if key in result:
            result[key] = "".join(
                [
                    (el1 if el1 == "0" else el2)
                    for el1, el2 in zip_longest(result[key], val, fillvalue=0)
                ]
            )
    return result
