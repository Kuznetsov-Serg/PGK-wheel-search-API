import asyncio
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from jose import jwt
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.auth import token_ldap, schemas
from app.settings import PARSED_CONFIG, TEMPLATES_PATH

# to include app api use next line
# from app.service_name.api.v1 import router as service_name_router
from app.utils.utils_template import my_url_for

auth_router = APIRouter(prefix=PARSED_CONFIG.api_prefix + "/auth", tags=["Authorization"])


def create_default_token():
    to_encode = {"iat": datetime.utcnow(), "exp": datetime.utcnow() + timedelta(days=2), "sub": "noAuthorised"}
    return jwt.encode(to_encode, PARSED_CONFIG.jwt.jwt_secret, algorithm=PARSED_CONFIG.jwt.jwt_algorithm)


if PARSED_CONFIG.is_authorise_by_token:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl=PARSED_CONFIG.root_path + auth_router.prefix + "/token/")
else:
    oauth2_scheme = create_default_token


templates = Jinja2Templates(directory=TEMPLATES_PATH)
templates.env.globals["my_url_for"] = my_url_for


@auth_router.post("/token/", name="Main authorization through LDAP (return Token)", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, token_ldap.ldap_check, form_data, db),
    )
    token_data_dict = {"sub": form_data.username}
    if PARSED_CONFIG.ad_authorize.is_enable:
        token_data_dict["ad_restriction"] = token_ldap.ldap_get_restriction(form_data.username)
    access_token = token_ldap.create_token(data=token_data_dict)
    token_ldap.write_user_history(db, form_data.username, form_data.password, "Successful registration")
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.get("/get-user/", response_model=schemas.User)
async def get_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    return await token_ldap.get_current_user(db, token)


@auth_router.get(
    "/get-user-right/",
    name="Get rights of current user.",
    description="Get rights of current User (Role with restriction of AD Group).",
    response_model=dict,
)
async def get_user_right(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    user = await token_ldap.get_current_user(db, token)
    if PARSED_CONFIG.ad_authorize.is_enable:
        payload = jwt.decode(token, PARSED_CONFIG.jwt.jwt_secret, algorithms=[PARSED_CONFIG.jwt.jwt_algorithm])
        ad_restriction = payload.get("ad_restriction", None)
        if not ad_restriction:
            ad_restriction = token_ldap.ldap_get_restriction(user.username)
    else:
        ad_restriction = None

    return token_ldap.get_user_access_dict(db, (user.id if user else 0), False, ad_restriction)


@auth_router.get(
    "/auth",
    name="Return simple HTML for insert username and password.",
    include_in_schema=False,
    response_class=HTMLResponse,
)
def auth(request: Request):
    return templates.TemplateResponse("auth_page.html", context={"request": request})


@auth_router.post("/auth", name="Авторизация через LDAP", include_in_schema=False, response_model=schemas.Token)
async def auth_ldap(user_auth_model: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, token_ldap.ldap_check, user_auth_model, db),
    )
    access_token = token_ldap.create_token()
    token_ldap.write_user_history(db, user_auth_model.username, user_auth_model.password, "Successful registration")
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.get("/", include_in_schema=False, response_class=HTMLResponse)
def secure_page(
    request: Request,
):
    return templates.TemplateResponse(
        "secure_page.html",
        context={
            "request": request,
        },
    )


@auth_router.get("/start_page", include_in_schema=False, response_class=HTMLResponse)
async def start_page(
    request: Request,
    # manager: NLMKManager = Depends(NLMKManager),
    # decoded_token=Depends(crud.check_token)
):
    return templates.TemplateResponse(
        "welcome.html",
        # 'start_page.html',
        context={
            "request": request,
            # 'mail_list': await manager.create_mailing_list(),
            # 'date_mail_list': await manager.unique_email_date_list()
        },
    )
