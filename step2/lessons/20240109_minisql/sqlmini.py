from sqlalchemy import *
from sqlalchemy.orm import registry

engine = create_engine("sqlite+pysqlite:///memory", echo=True)

metadata = MetaData()
user_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", BigInteger, unique=True),
    Column("full_name", String),
)

email_address = Table(
    "email_address",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey('users.user_id')),
    Column("email", String),
    Column("login", String, unique=True)
)

real_mail = Table(
    'real_mail',
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey('users.user_id')),
    Column("real_street", String),
    Column("house_num", Integer),
    Column("under_drive_num", Integer),
    Column("apart_num", Integer),
)

enter = Table(
    'enter_ATB',
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", BigInteger, unique=True),
    Column('login', ForeignKey('email_address.login')),
    Column('password', String(11))
)


metadata.create_all(engine)
metadata.drop_all(engine)