from app.auth import models, schemas
from app.auth.crud import UserCRUD, RoleItemCRUD, UserRoleCRUD
from app.utils.utils_crud import UniversalCRUD

routers_lst = []

# Authorisation
user_CRUD = UserCRUD(
    models.User,
    routers_lst,
    schemas.user_config_CRUD,
    schema_create=schemas.UserCreate,
    schema=schemas.User,
)
user_role_CRUD = UserRoleCRUD(
    models.UserRole,
    routers_lst,
    schemas.user_role_config_CRUD,
    schema_create=schemas.UserRoleCreate,
    schema=schemas.UserRole,
)
role_CRUD = UniversalCRUD(
    models.Role,
    routers_lst,
    schemas.role_config_CRUD,
    schema_create=schemas.RoleCreate,
    schema=schemas.Role,
)
role_item_CRUD = RoleItemCRUD(
    models.RoleItem,
    routers_lst,
    schemas.role_item_config_CRUD,
    schema_create=schemas.RoleItemCreate,
    schema=schemas.RoleItem,
    schema_update=schemas.RoleItemUpdate,
)
