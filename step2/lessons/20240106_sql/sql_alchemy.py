from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, Select
from sqlalchemy.orm import declarative_base, Session


engine = create_engine('sqlite:///memory.db', echo=True)
metadata = MetaData(bind=engine)

mytable = Table('mytable', metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String)
                )

metadata.create_all(bind=engine)
