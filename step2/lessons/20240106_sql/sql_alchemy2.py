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
    email = Column(String)  # Додано поле для електронної пошти

Base.metadata.create_all(bind=engine)

# Використовуйте контекстний менеджер для сесії
with Session(engine) as session:
    session.add_all([
        User(name="John", age=30),
        User(name="Alice", age=28),
        User(name="Bob", age=22)
    ])
    session.commit()

    # Відомості про користувачів з віком більше 25 років
    result = session.execute(select(User).where(User.age > 25))
    user_over_25 = result.scalars().all()

    print("Користувачі з віком більше 25 років:")
    for user in user_over_25:
        print(f"ID: {user.id}, Ім'я: {user.name}, Вік: {user.age}")

    # Додавання нового поля для електронної пошти
    email_column = Column('email', String)
    email_column.create(User.__table__)  # Використовуйте create для додавання колонки до таблиці

    # Оновлення електронних адрес для існуючих користувачів
    session.query(User).filter(User.name.in_(['John', 'Alice'])).update({User.email: 'example@email.com'},
                                                                        synchronize_session=False)
    session.commit()

    # Видалення користувачів з віком менше або дорівнює 30 рокам
    session.query(User).filter(User.age <= 30).delete()
    session.commit()

    # Запит для отримання всіх користувачів з їхніми електронними адресами
    result = session.execute(select(User))
    users_with_email = result.scalars().all()

    print("\nКористувачі з електронними адресами:")
    for user in users_with_email:
        print(f"id: {user.id}, Ім'я: {user.name}, Вік: {user.age}, Електронна пошта: {user.email}")
