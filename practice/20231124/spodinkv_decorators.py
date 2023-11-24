"""DECORATORS"""

"""
Task 1. Реалізуйте декоратор type_check, який перевіряє відповідність типів аргументів
функції заданим типам і викидає виняток, якщо типи не збігаються.
"""


def check_types(*types):
    def func_main(func):
        def wrapper(*args, **kwargs):
            for param, param_type in zip(args, types):
                if not isinstance(param, param_type):
                    raise TypeError(f'Expected {param_type.__name__}, but got {type(param).__name__} for argument {param}')
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


print(multiplied_text(10, 'Hello '))
# print(multiplied_text('World', 'Hello '))

"""
Task 2. Реалізуйте декоратор delay, який затримує виконання функції на вказану кількість секунд.
Функція повинна на заданий час давати shutdown, також повинна бути перевірка на статус код
та try/except для перевірки connection.
"""