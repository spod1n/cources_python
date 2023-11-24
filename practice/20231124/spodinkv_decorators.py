"""DECORATORS"""

"""
Task 1. Реалізуйте декоратор type_check, який перевіряє відповідність типів аргументів
функції заданим типам і викидає виняток, якщо типи не збігаються.
"""


def check_types(*types):
    """ Декоратор для перевірки типів змінних функції.

    :param types: (args) типи для звірки
    :return: результат обгорнутої функції
    """
    def func_main(func):
        """ Виклик функції, яку обгортаємо.

        :param func: назва функції, яку обгортаємо
        :return: результат обгортки
        """
        def wrapper(*args, **kwargs):
            """ Обгортка.

            Витягуємо по одному елементу з кожного зі списку args і types і об'єднує їх у кортежі.
            Перебираємо кожну пару (arg, arg_type), де arg - це аргумент функції, arg_type - відповідний тип.
            При кожній ітерації перевіряємо, чи arg є екземпляром arg_type.

            Якщо це не так, виникає TypeError, і виводиться повідомлення про помилку,
            що містить ім'я очікуваного типу, отриманого типу та сам аргумент.

            Якщо ітерація закінчилася без викиду вийнятку - виводимо інформацію, що перевірка типів відбулася успішно,
            та повертаємо результат обгорнутої функції.

            :return: результат обгорнутої функції
            """
            for param, param_type in zip(args, types):
                if not isinstance(param, param_type):
                    raise TypeError(f"Expected '{param_type.__name__}', "
                                    f"but got '{type(param).__name__}' for argument '{param}'")
            else:
                print('Check function types - OK!', end='\n\n')
                result = func(*args, **kwargs)
                return result
        return wrapper
    return func_main


@check_types(int, str)
def multiplied_text(num: int, text: str) -> str:
    """Функція для виведення тексту декілька разів.

    :param num: (int) кількість разів виведення тексту
    :param text: (str) текст для виведення
    :return: str
    """
    return text * num


# print(multiplied_text(10, 'Hello '))
print(multiplied_text('10', 'Hello '))


"""
Task 2. Реалізуйте декоратор delay, який затримує виконання функції на вказану кількість секунд.
Функція повинна на заданий час давати shutdown, також повинна бути перевірка на статус код
та try/except для перевірки connection.
"""

import time
import requests

from requests import HTTPError


def delay(shutdown_time: int):
    """ Декоратор для затримки виконання функції на вказану кількість секунд.

    :param shutdown_time: (int) кількість секунд для очікування
    :return: результат обгорнутої функції
    """
    def func_main(func):
        def wrapper(*args, **kwargs):
            time.sleep(shutdown_time)
            return func(*args, **kwargs)
        return wrapper
    return func_main


@delay(10)
def get_response(url: str):
    """ Виклик URL, перевірка коду відповіді API.
    Якщо код відповіді знаходиться не в діапазоні 200 - 299: викликаємо URL повторно.
    Кількість спроб виклику - 5.

    :param url: (str) адреса
    :return: (dict) відповідь API
    """
    for attmpt in range(1, 6):
        try:
            response = requests.get(url, verify=False)

            if (response.status_code >= 200) & (response.status_code < 300):
                return response.json()
            else:
                print(f'API response: {response.status_code}.. {attmpt} out of 5.. URL: {url}')
        except HTTPError as http_err:
            print(f'Http response error: {http_err}.. {attmpt} out of 5.. URL: {url}')
        except Exception as exc:
            print(f'Global error: {exc} {attmpt} out of 5.. URL: {url}')
    else:
        return False


URL = 'https://api.monobank.ua/bank/currency'
data_currency = get_response(URL)
print(data_currency)
