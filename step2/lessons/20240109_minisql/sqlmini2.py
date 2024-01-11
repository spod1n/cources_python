import json
from sqlalchemy import create_engine, Column, MetaData, Table, Integer, String, ForeignKey
from sqlalchemy.orm import Session, as_declarative, declared_attr

engine = create_engine("sqlite+pysqlite:///memory.db", echo=True)


@as_declarative()
class AbstractModel:
    id = Column(Integer, autoincrement=True, primary_key=True)

    @classmethod
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()


class UserModel(AbstractModel):
    __tablename__ = 'users'
    user_id = Column(Integer, nullable=False)
    name = Column(String)
    full_name = Column(String)


class AddressModel(AbstractModel):
    __tablename__ = 'address'
    email = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey(f'users.id'))


with Session(engine) as session:
    with session.begin():
        AbstractModel.metadata.create_all(engine)
        user1 = UserModel(user_id=1, name='Jack', full_name='Jack Sparroy')
        session.add(user1)
