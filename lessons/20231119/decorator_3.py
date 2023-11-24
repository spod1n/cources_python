"""
# Task 3. Використати декоратор, який буде зберігати cache, реалізувати для чисел Фіббоначі
# * - додаткова зробити власний декоратор memoize, та використати декоратор lru_cache, який буде працювати швидше
"""


import time
from functools import *


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Час виконання {func.__name__}: {end_time - start_time} секунд")
        return result
    return wrapper


@timer
@cache
def fibonacci_cache(n):
    return n if n <= 1 else fibonacci_cache(n-1) + fibonacci_cache(n-2)


@timer
@lru_cache(maxsize=4)
def fibonacci_lru_cache(n):
    return n if n <= 1 else fibonacci_lru_cache(n-1) + fibonacci_lru_cache(n-2)


def memoize(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper


@timer
@memoize
def fibonacci_memorize(n):
    return n if n <= 1 else fibonacci_memorize(n-1) + fibonacci_memorize(n-2)


print(fibonacci_cache(100))
print(fibonacci_lru_cache(100))
print(fibonacci_memorize(100))
