import logging
import random
import sys
from datetime import date
from typing import Any, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.util import asyncio
from tqdm.asyncio import tqdm

from app.api.deps import get_db, get_engine, get_db_ora, get_engine_ora
from app.auth.token_ldap import check_token, write_user_history
from app.core import crud, models, schemas
from app.core.crud import StealCRUD, StorageCRUD, WheelSetCRUD, MountingTypeCRUD, WorkFilterCRUD
from app.settings import EXCEL_MEDIA_TYPE, PARSED_CONFIG, SourceType

# to include app api use next line
# from app.service_name.api.v1 import router as service_name_router
from app.utils.log_db import LogDB, MyLogTypeEnum, LogSchema
from app.utils.utils import is_excel_by_content_type
from app.utils.utils_crud import UniversalCRUD

routers_lst = []

router_other = APIRouter(prefix=PARSED_CONFIG.api_prefix, dependencies=[Depends(check_token)], tags=["Other EndPoints"])
router_test = APIRouter(prefix=PARSED_CONFIG.api_prefix, tags=["Endpoints for test"])
# routers_lst.append(router_test)

logger = logging.getLogger()


# Дашборд возмещения убытков (Damage compensation)
steal_CRUD = StealCRUD(models.Steal, routers_lst, schemas.steal_config_CRUD, schemas.Steal, schemas.StealCreate)
work_filter_CRUD = WorkFilterCRUD(models.WorkFilter, routers_lst, schemas.work_filter_config_CRUD)
work_filter_storage_CRUD = UniversalCRUD(
    models.WorkFilterByStorage, routers_lst, schemas.work_filter_storage_config_CRUD
)
work_cost_CRUD = UniversalCRUD(models.WorkCost, routers_lst, schemas.work_cost_config_CRUD)
# check_filter_CRUD = CheckFilterCRUD(models.CheckFilter, routers_lst, schemas.check_filter_config_CRUD)

# Поиск колесных пар (Wheel Search)
wheel_set_CRUD = WheelSetCRUD(models.WheelSet, routers_lst, schemas.wheel_set_config_CRUD)
wheel_set_filter_CRUD = UniversalCRUD(models.WheelSetFilter, routers_lst, schemas.wheel_set_filter_config_CRUD)
wheel_set_cost_CRUD = UniversalCRUD(models.WheelSetCost, routers_lst, schemas.wheel_set_cost_config_CRUD)
mounting_type_CRUD = MountingTypeCRUD(models.MountingType, routers_lst, schemas.mounting_type_config_CRUD)
mounting_type_map_CRUD = UniversalCRUD(models.MountingTypeMap, routers_lst, schemas.mounting_type_map_config_CRUD)
storage_CRUD = StorageCRUD(models.Storage, routers_lst, schemas.storage_config_CRUD)

routers_lst.append(router_other)


@router_test.get("/swagger/", include_in_schema=False)
async def swagger_ui_html():
    """For add another swagger base url (/docs# ==> /api/swagger)"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=PARSED_CONFIG.project_name + " - Swagger UI",
        swagger_favicon_url="/static/images/favicon.png",
    )


# async def swagger_ui_html(req: Request) -> HTMLResponse:
#     """For add another swagger base url (/docs# ==> /api/swagger)"""
#     root_path = req.scope.get("root_path", "").rstrip("/")
#     # openapi_url = root_path + app.openapi_url
#     openapi_url = root_path + "/openapi.json"
#     # oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url
#     oauth2_redirect_url = '/docs/oauth2-redirect'
#     if oauth2_redirect_url:
#         oauth2_redirect_url = root_path + oauth2_redirect_url
#     return get_swagger_ui_html(
#         openapi_url=openapi_url,
#         title=PARSED_CONFIG.project_name + " - Swagger UI",
#         oauth2_redirect_url=oauth2_redirect_url,
#         # init_oauth=app.swagger_ui_init_oauth,
#         swagger_favicon_url="/static/images/favicon.png",
#         # swagger_ui_parameters=app.swagger_ui_parameters,
#     )


@router_test.get("/", include_in_schema=False)
def home():
    return RedirectResponse(PARSED_CONFIG.api_prefix + "/swagger/")


@router_test.get("/health-check/", name="A simple test that the application responds (live).")
def health_check() -> Any:
    return {"message": "OK"}


@router_test.get("/postgresql-check/", name="Test that the app has connected to PostgreSQL.")
def postgresql_check(db: Session = Depends(get_db), engine: Engine = Depends(get_engine)) -> Any:
    # df = pd.read_sql("select * from public.fact where date_rep between '2022-05-01' and '2022-05-31'", con=engine)
    # df.to_excel("fact_2022_05.xlsx", index=False)
    db.execute("SELECT 1 x")
    return {"message": "OK"}


@router_test.get("/ora-check/", name="Test that the app has connected to Oracle (Komandor).")
def oracle_check1(db: Session = Depends(get_db_ora)) -> Any:
    db.execute("SELECT 1 FROM dual")
    return {"message": "OK"}


@router_test.get(
    "/ora-get-row-check/", name="Test that the app has connected to Oracle (Komandor) and can get rows from Table."
)
async def oracle_check2(db: Session = Depends(get_db_ora)) -> Any:
    return db.execute(
        """
        SELECT st_code5, st_code6
        FROM (
            SELECT DISTINCT
                ROW_NUMBER() OVER(
                    PARTITION BY s.ST_CODE
                    ORDER BY s.RECDATEEND DESC
                ) rn
                , s.ST_CODE st_code5, s.ST_CODE6
            FROM nsi.STATION s
            WHERE s.ST_CODE IS NOT NULL
        ) WHERE rn = 1 and ROWNUM <= 10
        """
    ).fetchall()


@router_test.get("/sentry-check/", name="Checking the operation of the SENTRY trigger")
async def trigger_error():
    division_by_zero = 1 / 0


# @router.post(
#     "/load-work-spr/", name="Downloading the directory of analyzed types of work and details with prices (Excel-file)."
# )
async def work_spr_load(
    db: Session = Depends(get_db),
    engine: Engine = Depends(get_engine),
    uploaded_file: UploadFile = File(...),
    is_remove_current: bool = True,
    token=Depends(check_token),
) -> Any:
    if not is_excel_by_content_type(uploaded_file.content_type):
        raise HTTPException(
            status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
        )
    result = await crud.work_load(db, engine, uploaded_file, is_remove_current)
    # logger.info(f'User {PARSED_CONFIG.username} launched "load-filter-work-spr" (result={result})')
    write_user_history(
        db=db,
        username=token.get("sub", "NoAuthorised"),
        message=f'Called "{sys._getframe().f_code.co_name}" from file="{uploaded_file.filename}" ({result})',
    )
    return result


# @router.get("/work-filter-list/", response_model=list[schemas.WorkFilter])
async def work_filter_get_list(filter_by: str = "", skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    result = await work_filter_CRUD.get_list(db, filter_by, skip=skip, limit=limit)
    return result


# @router.get("/work-filter/{row_id}", response_model=schemas.WorkFilter)
async def work_filter_get(row_id: int, db: Session = Depends(get_db)):
    result = await work_filter_CRUD.get_record(db, row_id)
    return result


# @router.post("/work-filter/", response_model=schemas.WorkFilter)
async def work_filter_create(row: schemas.WorkFilterCreate = Depends(), db: Session = Depends(get_db)):
    username = PARSED_CONFIG.username
    result = await work_filter_CRUD.create_record(db, row)
    write_user_history(db=db, username=username, message=f'Called "{sys._getframe().f_code.co_name}"')
    return result


# @router.patch("/work-filter/{row_id}", response_model=schemas.WorkFilter)
async def work_filter_update(row_id: int, row: schemas.WorkFilterUpdate, db: Session = Depends(get_db)):
    result = await work_filter_CRUD.update_record(db, row_id, row)
    return result


# @router.delete("/work-filter/{row_id}")
async def work_filter_delete(row_id: int, db: Session = Depends(get_db)):
    result = await work_filter_CRUD.delete_record(db, row_id)
    return result


# @router.get("/work-filter-storage-list/", response_model=list[schemas.WorkFilterStorage])
async def work_filter_storage_get_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    result = await work_filter_storage_CRUD.get_list(db, skip=skip, limit=limit)
    return result


# @router.get("/work-filter-storage/{row_id}", response_model=schemas.WorkFilterStorage)
async def work_filter_storage_get(row_id: int, db: Session = Depends(get_db)):
    result = await work_filter_storage_CRUD.get_record(db, row_id)
    return result


# @router.post("/work-filter-storage/", response_model=schemas.WorkFilterStorage)
async def work_filter_storage_create(row: schemas.WorkFilterStorageCreate = Depends(), db: Session = Depends(get_db)):
    username = PARSED_CONFIG.username
    result = await work_filter_storage_CRUD.create_record(db, row)
    write_user_history(db=db, username=username, message=f'Called "{sys._getframe().f_code.co_name}"')
    return result


# @router.patch("/work-filter-storage/{row_id}", response_model=schemas.WorkFilterStorage)
async def work_filter_storage_update(row_id: int, row: schemas.WorkFilterStorageUpdate, db: Session = Depends(get_db)):
    result = await work_filter_storage_CRUD.update_record(db, row_id, row)
    return result


# @router.delete("/work-filter-storage/{row_id}")
async def work_filter_storage_delete(row_id: int, db: Session = Depends(get_db)):
    result = await work_filter_storage_CRUD.delete_record(db, row_id)
    return result


# @router.get("/work-cost-list/", response_model=list[schemas.WorkCost])
async def work_cost_get_list(filter_by: str = "", skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    result = await work_cost_CRUD.get_list(db, filter_by, skip=skip, limit=limit)
    return result


# @router.get("/work-cost/{row_id}", response_model=schemas.WorkCost)
async def work_cost_get(row_id: int, db: Session = Depends(get_db)):
    result = await work_cost_CRUD.get_record(db, row_id)
    return result


# @router.post("/work-cost/", response_model=schemas.WorkCost)
async def work_cost_create(row: schemas.WorkCostCreate = Depends(), db: Session = Depends(get_db)):
    username = PARSED_CONFIG.username
    result = await work_cost_CRUD.create_record(db, row)
    write_user_history(db=db, username=username, message=f'Called "{sys._getframe().f_code.co_name}"')
    return result


# @router.patch("/work-cost/{row_id}", response_model=schemas.WorkCost)
async def work_cost_update(row_id: int, row: schemas.WorkCostUpdate, db: Session = Depends(get_db)):
    result = await work_cost_CRUD.update_record(db, row_id, row)
    return result


# @router.delete("/work-cost/{row_id}")
async def work_cost_delete(row_id: int, db: Session = Depends(get_db)):
    result = await work_cost_CRUD.delete_record(db, row_id)
    return result


# @router.post("/steal-load/", name="Loading the fact of wagons repairs (excel-files)")
async def steal_load(
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
) -> Any:
    # checking for valid file types
    for uploaded_file in tqdm(files_list):
        if not is_excel_by_content_type(uploaded_file.content_type):
            raise HTTPException(
                status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
            )
    loop = asyncio.get_running_loop()
    username = PARSED_CONFIG.username
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
            is_append=False,
            username=username,
        )
        args = StealCRUD.load, db, engine, uploaded_file, period_load, is_overwrite, username, log_db, is_async
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


# @router.post("/check-spr-on-steal/", name="Checking WorkFilter for compliance with Steal (excel-files)")
async def check_spr_on_steal(
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
                None, crud.check_spr_on_steal, db, engine, uploaded_file, amount_difference, amount_repeat
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


# @router.post("/steal-list/", response_model=list[schemas.Steal])
# GET can't have BODY witch need for DICT ==> POST
async def steal_get_list(
    db: Session = Depends(get_db),
    date_from: Optional[date] = Query(None, title="date from", description="'Date from' (it is used if not empty)"),
    date_to: Optional[date] = Query(None, title="date to", description="'Date to' (it is used if not empty)"),
    is_include_date_null: Optional[bool] = Query(False, description="True - include rows with a 'date_detect' = Null"),
    # filter_model: dict = {
    #     "branch_name": {"filterType": "text", "type": "contains", "filter": "новосибирский филиал"},
    #     "wagon_num": {"filterType": "number", "type": "inList", "filter": [64561756, 64559610]},
    #     "is_insurance_of_carrier": {"type": "bool", "filter": "null"},
    # },
    # filter_model: dict = {"branch_name": "новосибирский филиал"},
    # filter_model: dict = dict(),
    filter_model: Optional[dict] = None,
    # sort_model=Body(
    #     [
    #         {"colId": "wagon_num", "sort": "asc"},
    #         {"colId": "branch_name", "sort": "desc"},
    #         {"colId": "storage_name", "sort": "asc"},
    #     ]
    # ),
    # sort_model=Body({"col_id": "date_detect", "sort": "desc"}),
    sort_model: Union[list, dict] = None,
    skip: int = 0,
    limit: int = 100,
):
    result = await steal_CRUD.get_list(
        db, date_from, date_to, is_include_date_null, filter_model, sort_model, skip, limit
    )
    return result


# @router.get("/steal/{row_id}", response_model=schemas.Steal)
async def steal_get(row_id: int, db: Session = Depends(get_db)):
    result = await steal_CRUD.get_record(db, row_id)
    return result

    # @router.post("/steal/", response_model=schemas.Steal)
    # async def steal_create(row: schemas.StealCreate = Body(), db: Session = Depends(get_db)):
    username = PARSED_CONFIG.username
    result = await steal_CRUD.create_record(db, row)
    write_user_history(db=db, username=username, message=f'Вызов "{sys._getframe().f_code.co_name}"')
    return result


# @router.get("/steal-generate-unique-id/")
async def steal_generate_unique_id(db: Session = Depends(get_db)):
    result = await steal_CRUD.generate_unique_id(db)
    return result


# @router.patch("/steal/{row_id}", response_model=schemas.Steal)
async def steal_update(row_id: int, row: schemas.StealUpdate, db: Session = Depends(get_db)):
    result = await steal_CRUD.steal_update(db, row_id, row)
    return result


# @router.delete("/steal/{row_id}")
async def steal_delete(row_id: int, db: Session = Depends(get_db)):
    result = await steal_CRUD.delete_record(db, row_id)
    return result


# @router.get("/steal-export/", name="Export steal (excel-file).")
async def steal_export(
    engine: Engine = Depends(get_engine),
    date_from: Optional[date] = Query(None, title="date from", description="'Date from' (it is used if not empty)"),
    date_to: Optional[date] = Query(None, title="date to", description="'Date to' (it is used if not empty)"),
    period_load: Optional[str] = Query(
        None,
        regex=r"^20[0-9]{2}-[0-9]{2}$",
        description="'Period load'='YYYY-MM' (used if not empty and the previous parameters are not specified)",
    ),
):
    file_steal = await steal_CRUD.export(engine, date_from, date_to, period_load)
    response = StreamingResponse(iter([file_steal.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
    d1 = date_from.strftime("%Y_%m_%d") if date_from else ""
    d2 = date_to.strftime("%Y_%m_%d") if date_to else ""
    file_name = f"steal_{d1}-{d2}.xlsx"
    response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response


# @router.post("/wheel-set-load/", name="Loading the Wheelset rejection from SAP/АСУ ВРК/Варрекс (excel-file).")
async def wheel_set_load(
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
        wheel_set_CRUD.load,
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


# @router.post("/wheel-set-list/", response_model=list[schemas.WheelSet])
# GET can't have BODY witch need for DICT ==> POST
async def wheel_set_get_list(
    db: Session = Depends(get_db),
    date_from: Optional[date] = Query(None, title="date from", description="'Date from' (it is used if not empty)"),
    date_to: Optional[date] = Query(None, title="date to", description="'Date to' (it is used if not empty)"),
    filter_model: Optional[dict] = None,
    sort_model: Union[list, dict] = None,
    skip: int = 0,
    limit: int = 100,
):
    # result = await wheel_set_CRUD.get_list_advanced(db, filter_model, sort_model, skip, limit)
    result = await wheel_set_CRUD.get_list(db, date_from, date_to, filter_model, sort_model, skip, limit)
    return result


# @router.get("/wheel-set/{row_id}", response_model=schemas.WheelSet)
async def wheel_set_get(row_id: int, db: Session = Depends(get_db)):
    result = await wheel_set_CRUD.get_record(db, row_id)
    return result


# @router.post("/wheel-set/", response_model=schemas.WheelSet)
async def wheel_set_create(row: schemas.WheelSetCreate = Body(), db: Session = Depends(get_db)):
    username = PARSED_CONFIG.username
    result = await wheel_set_CRUD.create_record(db, row)
    write_user_history(db=db, username=username, message=f'Вызов "{sys._getframe().f_code.co_name}"')
    return result


# @router.patch("/wheel-set/{row_id}", response_model=schemas.WheelSet)
async def wheel_set_update(row_id: int, row: schemas.WheelSetUpdate, db: Session = Depends(get_db)):
    result = await wheel_set_CRUD.update_record(db, row_id, row)
    return result


# @router.delete("/wheel-set/{row_id}")
async def wheel_set_delete(row_id: int, db: Session = Depends(get_db)):
    result = await wheel_set_CRUD.delete_record(db, row_id)
    return result


# @router.get("/wheel-set-export/", name="Export WheelSet (excel-file).")
async def wheel_set_export(
    engine: Engine = Depends(get_engine),
    date_from: Optional[date] = Query(None, title="date from", description="'Date from' (it is used if not empty)"),
    date_to: Optional[date] = Query(None, title="date to", description="'Date to' (it is used if not empty)"),
):
    file_steal = await wheel_set_CRUD.export(engine, date_from, date_to)
    response = StreamingResponse(iter([file_steal.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
    d1 = date_from.strftime("%Y_%m_%d") if date_from else ""
    d2 = date_to.strftime("%Y_%m_%d") if date_to else ""
    file_name = f"wheel_set_{d1}-{d2}.xlsx"
    response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response


# @router.get("/wheel-set-filter-list/", response_model=list[schemas.WheelSetFilter])
# async def wheel_set_filter_get_list(
#     filter_by: str = "", skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
# ):
#     result = await wheel_set_filter_CRUD.get_list(db, filter_by, skip=skip, limit=limit)
#     return result
#
#
# @router.get("/wheel-set-filter/{row_id}", response_model=schemas.WheelSetFilter)
# async def wheel_set_filter_get(row_id: int, db: Session = Depends(get_db)):
#     result = await wheel_set_filter_CRUD.get_record(db, row_id)
#     return result
#
#
# @router.post("/wheel-set-filter/", response_model=schemas.WheelSetFilter)
# async def wheel_set_filter_create(row: schemas.WheelSetFilterCreate = Depends(), db: Session = Depends(get_db)):
#     username = PARSED_CONFIG.username
#     result = await wheel_set_filter_CRUD.create_record(db, row)
#     write_user_history(db=db, username=username, message=f'Called "{sys._getframe().f_code.co_name}"')
#     return result
#
#
# @router.patch("/wheel-set-filter/{row_id}", response_model=schemas.WheelSetFilter)
# async def wheel_set_filter_update(row_id: int, row: schemas.WheelSetFilterUpdate, db: Session = Depends(get_db)):
#     result = await wheel_set_filter_CRUD.update_record(db, row_id, row)
#     return result
#
#
# @router.delete("/wheel-set-filter/{row_id}")
# async def wheel_set_filter_delete(row_id: int, db: Session = Depends(get_db)):
#     result = await wheel_set_filter_CRUD.delete_record(db, row_id)
#     return result
#
#
# @router.get("/wheel-set-filter-export/", name="Export Storage (excel-file).")
# async def wheel_set_filter_export(
#     db: Session = Depends(get_db), filter_by: str = "", skip: int = 0, limit: int = 99999
# ):
#     file_result = await wheel_set_filter_CRUD.export(db, filter_by, skip=skip, limit=limit)
#     response = StreamingResponse(iter([file_result.getvalue()]), media_type=EXCEL_MEDIA_TYPE)
#     file_name = f"{sys._getframe().f_code.co_name}_{date.today()}.xlsx"
#     response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
#     response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
#     return response
#
#
# @router.post("/wheel-set-cost-load/", name="Loading the WheelSelCost (excel-file).")
# async def wheel_set_cost_load(
#     uploaded_file: UploadFile = File(...),
#     is_overwrite: bool = False,
#     db: Session = Depends(get_db),
#     engine: Engine = Depends(get_engine),
# ) -> Any:
#     # checking for valid file types
#     if not is_excel_by_content_type(uploaded_file.content_type):
#         raise HTTPException(
#             status_code=422, detail=f'Недопустимый тип файла - ожидается "xlsx" (файл: "{uploaded_file.filename}")'
#         )
#     username = PARSED_CONFIG.username
#     result = await wheel_set_cost_CRUD.load(db, engine, uploaded_file, is_overwrite)
#     write_user_history(
#         db=db,
#         username=username,
#         message=f'Функцией "{sys._getframe().f_code.co_name}" обработан "{uploaded_file.filename}" ({result})',
#     )
#     return result


@router_other.get("/log-list/", response_model=list[LogSchema])
async def read_log_list(skip: int = 0, limit: int = 100, filter_parent_name: str = "", db: Session = Depends(get_db)):
    log_db = LogDB(db)
    log_list = log_db.get_list(filter_by=filter_parent_name, skip=skip, limit=limit)
    return log_list


@router_other.get("/log/", response_model=LogSchema)
async def read_log(
    log_id: Optional[int] = Query(None, title="ID log", description="ID log (it is used if not empty)"),
    parent_id: Optional[int] = Query(None, title="ID parent of log", description="it is used if log_id is empty"),
    parent_name: Optional[str] = Query(
        None, title="name parent of log", description='it is used if log_id is empty ("calc_tou", for example)'
    ),
    db: Session = Depends(get_db),
):
    log_db = LogDB(db)
    result = log_db.get(log_id=log_id, parent_id=parent_id, parent_name=parent_name)
    return result


@router_other.put("/log/", response_model=LogSchema)
async def log_write(
    log_id: Optional[int] = Query(None, title="ID log", description="ID log (it is used if not empty)"),
    parent_id: Optional[int] = Query(None, title="ID parent of log", description="it is used if log_id is empty"),
    parent_name: Optional[str] = Query(
        None, title="name parent of log", description='it is used if log_id is empty ("calc_tou", for example)'
    ),
    type_log: Optional[MyLogTypeEnum] = Query(None),
    msg: Optional[str] = Query("", title="message"),
    is_append: Optional[bool] = Query(True, description="True - add a message to the previous, False - overwrite"),
    is_with_time: Optional[bool] = Query(True, description="True - add datetime to begin msg"),
    db: Session = Depends(get_db),
):
    log_db = LogDB(db)
    kwargs = {
        "log_id": log_id,
        "parent_id": parent_id,
        "parent_name": parent_name,
        "type_log": type_log,
        "msg": msg,
        "is_with_time": is_with_time,
    }
    result = log_db.add(**kwargs) if is_append else log_db.put(**kwargs)
    return result
