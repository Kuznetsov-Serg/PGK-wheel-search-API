import inspect

from fastapi.params import Depends
from sqlalchemy import text
from sqlalchemy.orm import Session, DeclarativeMeta

from app.api.deps import get_db
from app.auth import models
from app.auth.token_ldap import check_token, write_user_history
from app.auth.utils import get_crud_type, get_user_access_dict
from app.settings import PARSED_CONFIG
from app.utils.utils_crud import UniversalCRUD


class UserCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-get-access-in-role/",
            response_model=dict,
            name="Get the user access rights specified in the roles.",
            description="Get the user access rights specified in the roles.",
            endpoint=self.endpoint_get_access_in_role,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-get-access/",
            response_model=dict,
            name="Get user access rights to all tables.",
            description="Get user access rights to all tables.",
            endpoint=self.endpoint_get_access,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )

    async def get_list_advanced(self,  db: Session, *args, **kwargs):
        result = await super().get_list_advanced(db, *args, **kwargs)
        # add the "role_list" field to the result
        if result:
            user_id_list = [el["id"] for el in result]
            sql_command = f'select user_id, role_id, "name" from user_role ur join "role" r on ur.role_id = r.id ' \
                          f'where user_id in ({", ".join(map(lambda x: f"{x}", user_id_list))})'
            user_role = db.execute(text(sql_command)).fetchall()
            for user in result:
                user["role_list"] = [{"id": el.role_id, "name": el.name} for el in user_role if el.user_id == user["id"]]
        return result

    async def endpoint_get_access_in_role(self, row_id: int, db: Session = Depends(get_db)) -> dict:
        user = await self.get_record(db, row_id)
        return get_user_access_dict(db, row_id)

    async def endpoint_get_access(self, row_id: int, db: Session = Depends(get_db)) -> dict:
        user = await self.get_record(db, row_id)
        return get_user_access_dict(db, row_id, is_only_in_role=False)


class RoleItemCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-crud-type/",
            response_model=dict,
            name="Get a list of CRUD types in a project",
            description="Get a list of CRUD types in a project (stored in RoleItem.access).",
            endpoint=self.endpoint_get_crud_type,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-table-type/",
            response_model=list[str],
            name="Get a list of tables in the project tracked for authorization.",
            description="Get a list of tables in the project tracked for authorization (stored in the RoleItem.item).",
            endpoint=self.endpoint_get_table_list,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )
        self.router.add_api_route(
            f"/{self.path_starts_with}-get-all-type/",
            response_model=dict,
            # response_model=list[str],
            name=f"Get all kinds of items.",
            description=f'Get all kinds of items from "{self.table_name}".',
            endpoint=self.endpoint_get_all_type,
            methods=["GET"],
            dependencies=[Depends(self.check_access_read)],
        )

    async def endpoint_get_all_type(
            self,
            db: Session = Depends(get_db),
            token=Depends(check_token),
    ) -> list[str]:
        result = self.get_tables_tracked_for_authorization_description()
        if PARSED_CONFIG.is_log_crud_all:
            write_user_history(
                db=db,
                username=token.get("sub", "NoAuthorised"),
                message=f'Просмотр всех видов таблиц, отслеживаемых по авторизации `{self.table_name}`'
            )
        return result

    @staticmethod
    async def endpoint_get_crud_type() -> dict:
        return get_crud_type()

    @staticmethod
    async def endpoint_get_table_list() -> list[str]:
        return PARSED_CONFIG.tables_tracked_for_authorization

    @staticmethod
    def get_tables_tracked_for_authorization_description() -> dict:
        for key, data in inspect.getmembers(models, inspect.isclass):
            if isinstance(data, DeclarativeMeta):
                if 'metadata' in data.__dict__ and 'tables' in data.metadata.__dict__:
                    return {table_name: table_obj.comment for table_name, table_obj in data.metadata.tables.items() if
                            table_name in PARSED_CONFIG.tables_tracked_for_authorization}
        return dict()


class UserRoleCRUD(UniversalCRUD):
    def _create_endpoints(self):
        super()._create_endpoints()
        self.router.add_api_route(
            f"/{self.path_starts_with}-update-list/",
            response_model=dict,
            name="Update (create + delete) a list of Roles for Users.",
            description="Update (create + delete) a list of Roles for Users.",
            endpoint=self.endpoint_update_list,
            methods=["PUT"],
            dependencies=[Depends(self.check_access_create), Depends(self.check_access_delete)],
        )

    async def endpoint_update_list(
            self,
            user_role_list: (list[dict] | None) = [{"user_id": 1, "role_id_list": [1, 2, 3]}],
            db: Session = Depends(get_db),
            token=Depends(check_token),
    ):
        if not user_role_list:
            return {"message": "Список изменений пуст."}

        # list of dict -> list of tuple
        user_role_new = []
        for el in user_role_list:
            for role_id in el["role_id_list"]:
                user_role_new.append((el["user_id"], role_id))
        user_role_new = set(user_role_new)

        user_id_list = [el["user_id"] for el in user_role_list]
        sql_command = f'select * from user_role where user_id in ({", ".join(map(lambda x: f"{x}", user_id_list))})'
        user_role = db.execute(text(sql_command)).fetchall()
        user_role_old = set([(el.user_id, el.role_id) for el in user_role])

        for_delete = user_role_old - user_role_new
        for_create = user_role_new - user_role_old

        if for_delete:
            id_for_delete = [el.id for el in user_role if (el.user_id, el.role_id) in for_delete]
            sql_command = f'delete from user_role where id in ({", ".join(map(lambda x: f"{x}", id_for_delete))})'
            amount_delete = db.execute(text(sql_command))
            db.commit()

        for el in for_create:
            await self.create_record(db, self.schema_create(**{"user_id": el[0], "role_id": el[1]}))

        message = f'Массовое изменение записей в таблице `{self.table_name}`:\n' \
                  f'{"удалены: " + str(for_delete) + " " if for_delete else ""}' \
                  f'{"добавлены: " + str(for_create) if for_create else ""}'

        if PARSED_CONFIG.is_log_crud_change and (for_delete & for_create):
            write_user_history(db=db, username=token.get("sub", "NoAuthorised"), message=message)
        return {"message": message}
