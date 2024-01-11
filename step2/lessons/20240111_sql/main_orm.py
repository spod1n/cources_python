from sqlalchemy import inspect, create_engine, MetaData, select, Column, Integer, String, ForeignKey, Table, insert
from sqlalchemy.orm import *
from sqlalchemy.dialects import *

engine = create_engine("sqlite+pysqlite:///memory.db", echo=True)

session = Session(engine, expire_on_commit=True)


class Base(DeclarativeBase):
    ...


class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    age: Mapped[int]


Base.metadata.create_all(engine)

user = User(id=1, name='test', age=30)
insp = inspect(user)
session.add(user)
print('is transient? ', insp.transient)
print('is pending? ', insp.pending)
session.flush()
print('is transient? ', insp.transient)
print('is pending? ', insp.pending)
print('is persistent? ', insp.persistent)
session.delete(user)
print('is deleted? ', insp.deleted)
session.flush()