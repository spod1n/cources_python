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

user = User(id=1, name='test1', age=30)
session.add(user)
session.flush()

session.expunge_all()

session_user = session.get(User, 1)
print(user == session_user)
print(user is session_user)
session_user.age = 1
print(session_user.age)
