""" SQL """

import sqlite3
from faker import Faker


def printer_sql_table(result_name: str, table: [tuple]) -> None:
    """ Функція для виведення даних

    :param result_name: (str) назва таблиці
    :param table: (list(tuple())) таблиця з sqlite3
    :return: None
    """
    print(result_name)

    for row in table:
        print(row)
    else:
        print('')


def create_table(conn: sqlite3.Connection) -> None:
    """ Функція для створення таблиці.

    :param conn: конектор sqlite3
    :return: None
    """
    conn.execute('''CREATE TABLE IF NOT EXISTS Person (id INTEGER PRIMARY KEY,
                                                       name TEXT,
                                                       surname TEXT,
                                                       date_of_birth DATE);''')
    conn.commit()


def insert_rows(conn: sqlite3.Connection, cur: sqlite3.Cursor, fake42: Faker) -> None:
    """ Функція для вставки даних в таблицю БД.

    :param conn: конектор sqlite3
    :param cur: курсор конектору
    :param fake42: дані для вставки
    :return: None
    """
    cur.execute(''' SELECT COUNT(1) FROM Person; ''')
    if not cur.fetchall()[0][0]:
        for i in range(1, 16):
            cur.execute('''INSERT INTO Person (id, name, surname, date_of_birth) VALUES (?, ?, ?, ?)''',
                        (i, fake42.first_name(), fake42.last_name(), fake42.date_of_birth()))
        else:
            conn.commit()


def select_hbdays(cur: sqlite3.Cursor, date1: str, date2: str) -> None:
    """ Функція для пошуку людей за датою народження.

    :param cur: курсор конектору
    :param date1: дата народження людини
    :param date2: дата народження людини
    :return: None
    """
    cur.execute('''SELECT * FROM Person WHERE date_of_birth IN (?, ?);''', (date1, date2))
    printer_sql_table("Happy Birthday's:", cur.fetchall())


def oldest_people(cur: sqlite3.Cursor):
    """ Функція для відображення найстарших людей.

    :param cur: курсор конектору
    :return: None
    """
    cur.execute('''SELECT * FROM Person ORDER BY date_of_birth ASC LIMIT 5;''')
    printer_sql_table('The oldest people:', cursor.fetchall())


def name_greater_surname(cur: sqlite3.Cursor):
    """ Функція для порівняння довжини імені з призвіща людини.

    :param cur: курсор конектору
    :return: None
    """
    cur.execute('''SELECT * FROM Person WHERE LENGTH(name) > LENGTH(surname)''')
    printer_sql_table('People with name greater than surname:', cur.fetchall())


DB_NAME = 'my_database.db'

with sqlite3.connect(DB_NAME) as connection:
    cursor = connection.cursor()

    """Task 1. create table Person, with id: int, primary key, name: text, surname: text, date_of_birth date"""
    create_table(connection)

    """Task 2. Заповнити таблицю Person 15 кортежі, *use - faker, *use - context manager"""
    fake = Faker()
    Faker.seed(42)
    insert_rows(connection, cursor, fake)

    """Task 3. Знайти всіх людей у кого день народження 19-03-1998 або 12-10-1998"""
    select_hbdays(cursor, '1998-03-19', '1998-10-12')

    """Task 4. Знайти найстарших людей в наборі (топ 5)"""
    oldest_people(cursor)

    """Task 5. Вивести всіх у кого name більше surname """
    name_greater_surname(cursor)
