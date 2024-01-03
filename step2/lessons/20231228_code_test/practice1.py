"""
Завдання 2: Тестування обробки помилок.

Напишіть функцію, яка кидає виняток за певних умов.
Напишіть тести, використовуючи pytest, щоб переконатися, що виняток правильно обробляється.
"""

import pytest


def is_num(num):
    if isinstance(num, int) and num < 0:
        raise ValueError('Number is smallest 1')
    return num ** 2


def test_1():
    # assert is_num(-1) == 1
    # assert is_num(0) == 0
    assert is_num(10) == 100

    with pytest.raises(ValueError):
        is_num(-1)


if __name__ == '__main__':
    test_1()
