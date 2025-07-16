from loguru import logger
from datetime import datetime, timedelta
from enum import Enum

import ldap
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from more_itertools import first
from passlib.context import CryptContext
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.auth import models, schemas
from app.auth.models import TypeCRUD
from app.auth.router import oauth2_scheme
from app.auth.utils import get_user_access_dict, get_all_access_false, restriction_sum
from app.settings import (
    PARSED_CONFIG,
    get_tables_user_tracked_for_authorization,
    get_tables_admin_tracked_for_authorization, get_restriction_for_user, get_restriction_for_admin,
    get_restriction_for_security_admin,
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# logger = logging.getLogger()


def ldap_auth(username, password):
    conn = ldap.initialize(f"ldap://{PARSED_CONFIG.ldap_server}")
    conn.protocol_version = 3
    conn.set_option(ldap.OPT_REFERRALS, 0)
    try:
        conn.simple_bind_s(f"{username}@pgk.rzd", password)
    except ldap.INVALID_CREDENTIALS:
        return {"status": "error", "message": "Неверный логин или пароль"}
    except ldap.SERVER_DOWN:
        return {"status": "error", "message": "Сервер авторизации недоступен"}
    except ldap.LDAPError as e:
        if type(e.message) == dict and e.message.has_key("desc"):
            return {"status": "error", "message": "Other LDAP error: " + e.message["desc"]}
        else:
            return {"status": "error", "message": "Other LDAP error: " + e}
    finally:
        conn.unbind_s()

    if PARSED_CONFIG.ad_authorize.is_enable and ldap_get_ad_group(username) == []:
        return {"status": "error", "message": f"Пользователь не входит в группы проекта (AD)."}

    return {"status": "ok", "message": "Успешно"}


def create_token(data: dict = None, *, expires_delta: timedelta = None):
    iat = datetime.utcnow()
    if expires_delta:
        expire = iat + expires_delta
    else:
        expire = iat + timedelta(days=PARSED_CONFIG.jwt.jwt_access_token_days)

    to_encode = {"iat": iat, "exp": expire}

    if data is not None:
        to_encode |= data

    encoded_jwt = jwt.encode(to_encode, PARSED_CONFIG.jwt.jwt_secret, algorithm=PARSED_CONFIG.jwt.jwt_algorithm)
    return encoded_jwt


def check_token(token: str = Depends(oauth2_scheme), exc_status=status.HTTP_401_UNAUTHORIZED):
    credentials_exception = HTTPException(
        status_code=exc_status,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token_decoded = jwt.decode(token, PARSED_CONFIG.jwt.jwt_secret, algorithms=[PARSED_CONFIG.jwt.jwt_algorithm])
        PARSED_CONFIG.username = token_decoded.get("sub", "NoAuthorised")
        # print(f"check_token token={token}\ntoken_decoded={token_decoded}")
        return token_decoded
    except Exception as exc:
        # except jwt.JWTError as err:
        # except jwt.InvalidTokenError as exc:
        raise credentials_exception


def ldap_check(user_auth_model: OAuth2PasswordRequestForm, db: Session):
    if PARSED_CONFIG.is_authorise_by_token:
        if is_login_enable(db, user_auth_model.username):
            pgk_cred = ldap_auth(user_auth_model.username, user_auth_model.password)
            if pgk_cred["status"] == "error":
                write_user_login_history(db, user_auth_model.username, False)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=pgk_cred["message"],
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                write_user_login_history(db, user_auth_model.username, True)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authorization was blocked for {PARSED_CONFIG.fail_login_period_block} minutes "
                       f"due to {PARSED_CONFIG.fail_login_try} failed attempts.",
                headers={"WWW-Authenticate": "Bearer"},
            )


async def check_access(db: Session, table_name: str, type_crud: (str | TypeCRUD), username: str = None):
    if PARSED_CONFIG.is_authorise_by_token and PARSED_CONFIG.is_authorise_by_role:
        username = username if username else PARSED_CONFIG.username
        result = db.query(models.User).filter(func.lower(models.User.username) == username.lower()).first()
        if result and not result.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"The user '{username}' is not active.",
            )
        access = get_user_access_dict(db, (result.id if result else 0), False)
        if table_name.lower() in access:
            type_crud = type_crud.value if isinstance(type_crud, Enum) else type_crud
            if not access[table_name.lower()].get(type_crud, False):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"The user '{username}' was not granted the rights of '{type_crud}' to '{table_name}'.",
                )


class CheckAccess:
    def __init__(self, table_name: str, type_crud: (str | TypeCRUD)):
        self.table_name = table_name.lower()
        self.type_crud = type_crud.value if isinstance(type_crud, Enum) else type_crud.lower()

    def __call__(self, token=Depends(check_token), db: Session = Depends(get_db)):
        if PARSED_CONFIG.is_authorise_by_token and PARSED_CONFIG.is_authorise_by_role:
            username = token.get("sub", "NoAuthorised")
            result = db.query(models.User).filter(func.lower(models.User.username) == username.lower()).first()
            if result and not result.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"The user '{username}' is not active.",
                )

            if PARSED_CONFIG.ad_authorize.is_enable:
                ad_restriction = token.get("ad_restriction", None)
                if not ad_restriction:
                    ad_restriction = ldap_get_restriction(username)
            else:
                ad_restriction = None

            access = get_user_access_dict(db, (result.id if result else 0), False, ad_restriction)
            if self.table_name in access:
                if not access[self.table_name].get(self.type_crud, False):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"The user '{username}' was not granted the rights of '{self.type_crud}' to '{self.table_name}'.",
                    )


async def get_current_user(db: Session, token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, PARSED_CONFIG.jwt.jwt_secret, algorithms=[PARSED_CONFIG.jwt.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_or_create_user(db, token_data.username)
    if user is None:
        raise credentials_exception
    return user


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_or_create_user(db: Session, username, password: str = "NULL"):
    result = db.query(models.User).filter(models.User.username == username.lower()).first()
    return result if result else create_user(db, username, password)


def create_user(db: Session, username: str, password: str):
    try:
        any_user = db.query(models.User).filter(models.User.is_active).first()
        db_result = models.User(
            username=username.lower(), email=username + "@pgk.ru", hashed_password=encrypt_password(password)
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        # created the first user - we will grant all rights
        if not any_user:
            gives_all_roles_to_user(db, db_result.id)
    except Exception as err:
        raise HTTPException(status_code=409, detail=f"Error: {err}")
    return db_result


def gives_all_roles_to_user(db: Session, user_id: int):
    try:
        result = get_or_create_all_roles(db)
        for role in result:
            db_result = models.UserRole(user_id=user_id, role_id=role.id)
            db.add(db_result)
        db.commit()
    except Exception as err:
        raise HTTPException(status_code=409, detail=f"Error: {err}")
    return db_result


def get_or_create_all_roles(db: Session):
    def add_role(name: str) -> models.Role:
        db_role = models.Role(name=name)
        db.add(db_role)
        db.commit()
        db.refresh(db_role)
        return db_role

    def add_role_item(role_id: int, tables: list, is_only_read: bool = False):
        access = "01" + "0" * (len(TypeCRUD) - 3) + "1" if is_only_read else "1" * len(TypeCRUD)
        for el in tables:
            result = models.RoleItem(role_id=role_id, item=el, access=access)
            db.add(result)
        db.commit()

    try:
        db_result = db.query(models.Role).all()
        if not db_result:  # there are no roles
            role = add_role("admin")
            add_role_item(role.id, get_tables_admin_tracked_for_authorization())
            role = add_role("user")
            add_role_item(role.id, get_tables_user_tracked_for_authorization())
            role = add_role("user_read_only")
            add_role_item(role.id, get_tables_user_tracked_for_authorization(), is_only_read=True)
            db_result = db.query(models.Role).all()
    except Exception as err:
        raise HTTPException(status_code=409, detail=f"Error: {err}")
    return db_result


def write_user_history(db: Session, username: str = None, password: str = "NULL", message: str = ""):
    try:
        username = first([username, PARSED_CONFIG.username, "NoAuthorised"])
        db_user = get_or_create_user(db, username, password)
        db_result = models.UserHistory(user_id=db_user.id, description=message)
        db.add(db_result)
        db.commit()
        logger.info(f'{str(datetime.now()).split(".", 2)[0]} - User {db_user.username} message="{message}"')
    except Exception as err:
        msg = f"Error in the adding process write_user_history " f"(user_name={username}): {err}"
        logger.error(msg)
        raise HTTPException(status_code=409, detail=msg)
    return db_result


def write_user_login_history(db: Session, username: str = None, is_success: bool = True):
    try:
        if not username:
            username = PARSED_CONFIG.username
        db_user = get_or_create_user(db, username)
        if not db_user:
            msg = f"Error in the adding process write_user_login_history (user_name={username}): " \
                  f"No such user has been found"
            logger.error(msg)
            raise HTTPException(status_code=409, detail=msg)

        db_result = models.UserLoginHistory(user_id=db_user.id, is_success=is_success)
        db.add(db_result)
        db.commit()
        logger.info(f"Write `user_login_history` for User {db_user.username} is_success=`{is_success}`")
    except Exception as err:
        msg = f"Error in the adding process write_user_login_history (user_name={username}): {err}"
        logger.error(msg)
        raise HTTPException(status_code=409, detail=msg)
    return db_result


def is_login_enable(db: Session, username: str = None) -> bool:
    result = False
    if not username:
        username = PARSED_CONFIG.username
    db_user = get_user_by_username(db, username)
    if db_user:
        db_result = (
            db.query(models.UserLoginHistory)
            .filter(
                models.UserLoginHistory.user_id == db_user.id,
                models.UserLoginHistory.date > datetime.now() - timedelta(minutes=PARSED_CONFIG.fail_login_period_block),
            )
            .order_by(models.UserLoginHistory.date.desc())
            .limit(PARSED_CONFIG.fail_login_try + 1)
            .all()
        )
        fail_login_try = sum([1 for el in db_result if not el.is_success])
        if fail_login_try <= PARSED_CONFIG.fail_login_try:
            result = True
    else:
        # There are no users
        if not db.query(models.User).filter(models.User.is_active).first():
            result = True
    return result


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(func.lower(models.User.username) == username.lower()).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def verify_password(plain_password, hashed_password):
    # print('plain_password=', plain_password)
    # print('hashed_password=', hashed_password)
    # hashed_password = get_password_hash(plain_password)
    # print(f'verify={pwd_context.verify(plain_password, hashed_password)} get_password_hash={hashed_password}')
    # hashed_password = encrypt_password(plain_password)
    # print(f'verify={pwd_context.verify(plain_password, hashed_password)} encrypt_password={hashed_password}')
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def encrypt_password(password):
    return pwd_context.encrypt(password)


def authenticate_user(db: Session, username: str, password: str):
    user = get_or_create_user(db, username, password)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def ldap_get_user_name_by_email_from_ad(email: str) -> dict:
    # Setting the necessary variables
    base = PARSED_CONFIG.ldap_dc
    bind_dn = PARSED_CONFIG.ad_authorize.user
    bind_dn_password = PARSED_CONFIG.ad_authorize.password
    scope = ldap.SCOPE_SUBTREE
    filter = f"(&(mail={email}))"
    attrs = ['name', 'mail', 'proxyAddresses', 'title', 'department', 'description', 'info', 'company', 'streetAddress',
             'manager', 'l', 'memberOf']
    # attrs = ['*']
    # Connecting to the global directory via LDAP
    try:
        conn = ldap.initialize(f"ldap://{PARSED_CONFIG.ldap_server}")
        conn.protocol_version = ldap.VERSION3
        conn.set_option(ldap.OPT_REFERRALS, 0)
        conn.simple_bind_s(bind_dn, bind_dn_password)
        # except ldap.SERVER_DOWN:
        # except Exception as err:
        #     raise HTTPException(status_code=409, detail=f"Error: {err}")

        # Get the results of the search for objects in AD
        ldap_result_id = conn.search_ext(base, scope, filter, attrs)
    # try:
        while 1:
            result_type, result_data = conn.result(ldap_result_id, 0)
            if not result_data:
                return None
            if result_type == ldap.RES_SEARCH_ENTRY:
                result = [el for el in result_data[0][0].split(",") if el.startswith("CN=")]
                if result:
                    attrs = list(result_data[0][1].keys()) if attrs == ["*"] else attrs
                    return {el: result_data[0][1][el][0].decode("utf_8", errors='replace') if el in result_data[0][
                        1].keys() else None for el in attrs}
                else:
                    return None
    except Exception as err:
        raise HTTPException(status_code=409, detail=f"Error: {err}")


def ldap_email_in_ad(email_list: list[str], is_http_error: bool = True) -> list:
    try:
        # Connecting to the global directory via LDAP
        conn = ldap.initialize(f"ldap://{PARSED_CONFIG.ldap_server}")
        conn.protocol_version = ldap.VERSION3
        conn.set_option(ldap.OPT_REFERRALS, 0)
        conn.simple_bind_s(PARSED_CONFIG.email.user, PARSED_CONFIG.email.password)

        result_list = []
        for email in email_list:
            # Get the results of the search for objects in AD
            ldap_result_id = conn.search_ext(
                PARSED_CONFIG.ldap_dc,
                ldap.SCOPE_SUBTREE,
                f"(&(mail={email}))",
                ['mail', 'proxyAddresses'],
            )
            while 1:
                result_type, result_data = conn.result(ldap_result_id, 0)
                if result_data:
                    if result_type == ldap.RES_SEARCH_ENTRY:
                        result_list.append(email)
                        break
                else:
                    break
        return list(set(result_list))

    except Exception as err:
        raise HTTPException(status_code=409, detail=f"Error: {err}") if is_http_error else err


def ldap_get_all_email_from_ad():
    # Setting the necessary variables
    base = PARSED_CONFIG.ldap_dc
    bind_dn = PARSED_CONFIG.email.user
    bind_dn_password = PARSED_CONFIG.email.password
    scope = ldap.SCOPE_SUBTREE
    # filter = "(&(mail=kuznetsovsn@pgk.ru))"
    filter = "(&(mail=*))"
    attrs = ['mail', 'proxyAddresses']
    result_set = []
    all_emails = []

    # Connecting to the global directory via LDAP
    try:
        conn = ldap.initialize(f"ldap://{PARSED_CONFIG.ldap_server}")
        conn.protocol_version = ldap.VERSION3
        conn.set_option(ldap.OPT_REFERRALS, 0)
        conn.simple_bind_s(bind_dn, bind_dn_password)
    except ldap.SERVER_DOWN:
        print("Error connection to AD")

    # Get the results of the search for objects in AD
    ldap_result_id = conn.search_ext(base, scope, filter, attrs)
    try:
        while 1:
            result_type, result_data = conn.result(ldap_result_id, 0)
            if result_data:
                if result_type == ldap.RES_SEARCH_ENTRY:
                    result_set.append(result_data)
            else:
                break
    except ldap.SIZELIMIT_EXCEEDED:
        print("The limit on the number of records received from AD has been reached")

    # get a list of email addresses and put it in the 'all_emails'
    for user in result_set:
        proxy_addresses = user[0][1].get('proxyAddresses')
        mail = user[0][1].get('mail')
        if proxy_addresses:
            for email_b in proxy_addresses:
                email = email_b.decode("utf-8")
                all_emails.append(email.split(':')[1])
        else:
            all_emails.append(mail[0].decode("utf-8"))

    unique_all_emails = list(set(all_emails))
    print(*unique_all_emails, sep='\n')


def ldap_get_restriction(username: str) -> dict:
    if not PARSED_CONFIG.ad_authorize.is_enable:
        return None

    result = get_all_access_false()
    try:
        group_list = ldap_get_ad_group(username.lower().strip())
        for group in group_list:
            if group == PARSED_CONFIG.ad_authorize.user_group.lower():
                result = restriction_sum(result, get_restriction_for_user())
            if group == PARSED_CONFIG.ad_authorize.admin_group.lower():
                result = restriction_sum(result, get_restriction_for_admin())
            if group == PARSED_CONFIG.ad_authorize.security_admin_group.lower():
                result = restriction_sum(result, get_restriction_for_security_admin())
    except Exception as err:
        logger.error(f"Error: {err}")
    finally:
        return result


def ldap_get_ad_group(username: str) -> list:
    def get_group(user_dict: dict) -> list:
        if user_dict and "memberOf" in user_dict.keys():
            group_list = [el[el.find("=")+1:].lower() for el in user_dict["memberOf"].split(",")]
            return [el for el in group_list if el in group_search]
        return []

    result = []
    try:
        if PARSED_CONFIG.ad_authorize.is_enable:
            group_search = [
                PARSED_CONFIG.ad_authorize.user_group.lower(),
                PARSED_CONFIG.ad_authorize.admin_group.lower(),
                PARSED_CONFIG.ad_authorize.security_admin_group.lower(),
            ]
            result = get_group(ldap_get_user_name_by_email_from_ad(f"{username.lower().strip()}@pgk.ru"))
            if not result:
                result = get_group(ldap_get_user_name_by_email_from_ad(f"{username.lower().strip()}@pgkdigital.ru"))
    except Exception as err:
        logger.error(f"Error: {err}")
    finally:
        return result
