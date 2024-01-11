from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, select
from sqlalchemy.orm import declarative_base, Session

engine = create_engine('sqlite:///memory.db')
metadata = MetaData()

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)


Base.metadata.create_all(bind=engine)

with Session(engine) as session:
    session.add_all([
        User(name="John", age=30),
        User(name="Alice", age=28),
        User(name="Bob", age=22)
    ])
    session.commit()

    result = session.execute(select(User).where(User.age > 25))
    user_over_25 = result.scalars().all()

    print("Користувачі з віком більше 25 років:")
    for user in user_over_25:
        print(f"ID: {user.id}, Ім'я: {user.name}, Вік: {user.age}")
