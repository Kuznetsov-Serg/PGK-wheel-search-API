from typing import Any

from pydantic import BaseModel

from app.utils.utils_schema import partial_model


class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True
        from_attributes = True


class Token(OurBaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: (str | None) = None


class RoleItemBase(OurBaseModel):
    role_id: int
    item: str
    access: str
    # role: Any


class RoleItemCreate(OurBaseModel):
    role_id: int
    item: str
    access_dict: dict


@partial_model
class RoleItemUpdate(OurBaseModel):
    role_id: int
    item: str
    # access: str
    access_dict: dict


class RoleItem(RoleItemBase):
    id: int
    access_dict: (dict | None)


class RoleBase(OurBaseModel):
    name: str


class RoleCreate(RoleBase):
    pass


class Role(RoleBase):
    id: int
    role_item: (list[RoleItem] | None)


class UserRoleBase(OurBaseModel):
    user_id: int
    role_id: int


class UserRoleCreate(UserRoleBase):
    pass


class UserRole(UserRoleBase):
    id: int
    # user: Any
    role: Role


class UserBase(OurBaseModel):
    username: str
    email: (str | None) = None
    is_active: (bool | None) = None
    # user_role: (list[UserRole] | None)


class UserCreate(UserBase):
    pass
    # hashed_password: str


class User(UserBase):
    id: int
    # access_dict: (dict | None)
    # hashed_password: (str | None) = None
    user_role_list: (list[int] | None)
    user_role: (list[UserRole] | None)


user_config_CRUD = {"filter_by": "username", "sort_model": [{"col_id": "username"}]}
user_role_config_CRUD = {"sort_model": [{"col_id": "user_id"}]}
role_config_CRUD = {"filter_by": "name", "sort_model": [{"col_id": "name"}]}
role_item_config_CRUD = {"filter_by": "item", "sort_model": [{"col_id": "item"}]}
