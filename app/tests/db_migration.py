import pandas as pd
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.orm import sessionmaker

from app.api.deps import get_db, get_engine

tables = [
    "wheel_set",
    "wheel_set_filter",
    "mounting_type",
    "mounting_type_map",
    "wheel_set_cost",
    "storage",
    "work_filter",
    "work_filter_by_storage",
    "work_cost",
    "steal",
    "check_filter",
    "role",
    "role_item",
    "user",
    "user_role",
    "user_history",
    "user_login_history",
    "log",
]
engine_from = next(get_engine())


def migrate():
    engine_to = create_engine(
        'postgresql://sliding_planning:HvyhDb4pi8rFbvxS@dp-postgres-dev.pgk.ru:5432/poc_sliding_planning',
        pool_pre_ping=False,
    )
    # engine_to = next(get_engine())
    # db_to = next(get_db())
    session_to = sessionmaker(autocommit=False, autoflush=False, bind=engine_to)
    db_to = session_to()


    for table in tables:
        query = f'SELECT * FROM "{table}"'
        df = pd.read_sql(query, engine_from)
        print(f"Reading from `{table}` {df.shape[0]} rec.")
        db_to.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
        db_to.commit()
        metadata = MetaData()
        table_new = Table(table, metadata, autoload_with=engine_from)
        table_new.create(engine_to)

        df.to_sql(table, engine_to, index=False, if_exists='append')

        if df.shape[0]:
            query = f"SELECT setval('{table}_id_seq',(SELECT GREATEST(MAX(id), nextval('{table}_id_seq')-1) FROM {table}))"
            result = db_to.execute(text(query))


def update_seq_for_primary_key():
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine_from)
    db = session()
    for table in tables:
        query = f"""
                    SELECT setval('{table}_id_seq',(SELECT GREATEST(MAX(id), nextval('{table}_id_seq')-1) FROM "{table}"))
                """
        result = db.execute(text(query))


update_seq_for_primary_key()