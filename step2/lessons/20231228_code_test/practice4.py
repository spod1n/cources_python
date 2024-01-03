"""
Завдання: Параметризовані випробування.

Напишіть параметризований тест, який приймає різні вхідні дані для функції та перевіряє очікуваний результат.
Використовуйте @pytest.mark.parametrize для передачі різних аргументів на тест.
"""
import pytest

def add(a,b):
    return a + b

@pytest.mark.parametrize('a, b, expected', [(2,3,5), (0,0,0), (-1,1,0)])
def test_a_p(a, b, expected):
    assert  add(a, b) == expected
    