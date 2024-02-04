import hashlib
import time
import re

import sqlite3


class User:
    # %%
    # magic
    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.email = email
        self.db_name = 'users.db'

    def __enter__(self):
        """ Підключаємось до БД sqlite3, створюємо курсор при виклику блоку 'with' """
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()

        # створюємо таблицю users, якщо її не існує в БД
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                email TEXT UNIQUE
            )''')

        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        """ Комітимо зміни та закриваємо БД sqlite3 при завершенні блоку 'with' """
        self.cursor.connection.commit()
        self.cursor.connection.close()

    # %%
    # main
    def register(self):
        """ Метод реєстрації: - перевірка на коректний email, - перевірка на складність пароля """
        if self.validate_email(self.email):
            if self.check_pass(self.password):
                self._register_true()
            else:
                while True:
                    user_input = input('Пароль занадто легкий. Ви впевнені, що бажаєте продовжити? (Y/N): ')
                    if user_input.lower() == 'y':
                        self._register_true()
                        break
                    elif user_input.lower() == 'n':
                        print('-' * 50)
                        print('Реєстрація не підтверджена..')
                        break
                    else:
                        print('-' * 50)
                        print('Неправильний вибір. Спробуйте ще раз.')

        else:
            print('-' * 50)
            print('Введено некоректний email. Реєстрація скасована..')

    def login(self) -> bool:
        """ Метод для входу. Перевірка логіну і паролю в БД """
        try:
            with self:
                self.cursor.execute('SELECT * FROM users WHERE username=? AND password=?',
                                    (self.username, self.password_hash))
                user = self.cursor.fetchone()

        except sqlite3.Error as exc_str:
            print('Error..', f'Помилка під час входу: {exc_str}', sep='\n')
            return False

        if user:
            print('-' * 50)
            print('Успішний вхід!')
            return True
        else:
            print('-' * 50)
            print("Ведено некоректне ім'я користувача або пароль!")
            return False

    # %%
    # internal
    def _register_true(self):
        """ Допоміжний метод реєстрації: - вставка користувача в БД, - перевірка на існування користувача """
        try:
            with self:
                self.cursor.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                                    (self.username, self.password_hash, self.email))
        except sqlite3.IntegrityError:
            print('-' * 50)
            print('Помилка!', "Користувач з таким ім'ям або електронною адресою вже існує..", sep='\n')
        else:
            print('-' * 50)
            print('Реєстрація успішна!')

    # %%
    # static
    @staticmethod
    def validate_email(email):
        """ Метод перевірки коректності електронної пошти """
        email_pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
        return True if re.match(email_pattern, email) else False

    @staticmethod
    def check_pass(password):
        if len(password) >= 5:
            if any(char.isupper() for char in password):
                if any(char.isdigit() for char in password):
                    return True
        return False


if __name__ == '__main__':
    while True:
        print('-' * 50)
        print('[1] Зареєструватися', '[2] Увійти', '[3] Вийти', sep='\n', end='\n')
        print('-' * 50)
        choice = input('Оберіть опцію: ')

        match choice:
            case '1':
                print('-' * 50)
                print('Реєстрація..')

                user_name = input("Введіть ім'я користувача: ")
                user_pass = input('Введіть пароль: ')
                user_email = input('Введіть електронну адресу: ')

                new_user = User(user_name, user_pass, user_email)
                new_user.register()

            case '2':
                print('-' * 50)
                print('Вхід..')

                user_name = input("Введіть ім'я користувача: ")
                user_pass = input('Введіть пароль: ')

                existing_user = User(user_name, user_pass, '')
                existing_user.login()

            case '3':
                print('-' * 50)
                print('Вихід з програми..')
                time.sleep(2)
                break

            case _:
                print('-' * 50)
                print('Неправильний вибір. Спробуйте ще раз.')
