import pandas as pd
import numpy as np

from sqlalchemy import create_engine, MetaData, select, Column, Integer, String, ForeignKey, Table, insert
from sqlalchemy.orm import Session, as_declarative, declared_attr
from sqlalchemy.dialects import *


engine = create_engine("sqlite+pysqlite:///memory.db", echo=True)

metadata = MetaData()

user_table = Table(
    'users',
    metadata,
    Column('id', Integer, primary_key=True, unique=True, autoincrement=True),
    Column('name', String(30)),
    Column('full_name', String(90))
)

address_table = Table(
    'address',
    metadata,
    Column('id', Integer, primary_key=True, unique=True, autoincrement=True),
    Column('email', String(100)),
    Column('user_id', ForeignKey('users.id'))
)

metadata.create_all(engine)

# stmt = insert(user_table).values(name='Test', full_name='Test Test')
stmt_no_values = insert(user_table)

# sqlite_stmt = stmt_no_values.compile(engine, sqlite.dialect())
mssql_stmt = stmt_no_values.compile(engine, mssql.dialect())

num_rows = 1_000_000
data = {
    'name': np.random.choice(['Petro', 'Ivan', 'Sergii'], num_rows),
    'full_name': np.random.choice(['Petro Sagaydachnij', 'Ivan Bogun', 'Sergii Nepiypivo'], num_rows)
}

df = pd.DataFrame(data)

with engine.begin() as conn:
    result = conn.execute(stmt_no_values, df.to_dict('records'))

# metadata.drop_all(engine)
