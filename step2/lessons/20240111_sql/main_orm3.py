import pandas as pd
import numpy as np

from sqlalchemy import inspect, create_engine, MetaData, select, Column, Integer, String, ForeignKey, Table, insert
from sqlalchemy.orm import *
from sqlalchemy.dialects import *

engine = create_engine("sqlite+pysqlite:///memory.db", echo=True)

session = Session(engine, expire_on_commit=True)


class Base(DeclarativeBase):
    ...


class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    full_name: Mapped[str]


Base.metadata.create_all(engine)

num_rows = 2000
data = {
    'name': np.random.choice(['Petro', 'Ivan', 'Sergii'], num_rows),
    'full_name': np.random.choice(['Petro Sagaydachnij', 'Ivan Bogun', 'Sergii Nepiypivo'], num_rows)
}

df = pd.DataFrame(data)


for i in df.to_dict('records'):
    user = User(name=i['name'], full_name=i['full_name'])
    session.add(user)
    session.flush()

    # for x in range(0, 10):
    #     user_from_db = session.scalar(select(User).where(User.id == x))

    # print(user_from_db == user)
    # print(user_from_db is user)

    session.commit()
    # session.delete(user_from_db)
