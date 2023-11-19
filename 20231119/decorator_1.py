"""
# Task 1. Написати декоратор, де буде показувати імя функції та дату
"""


import datetime


def func_and_now(func):
    def wrapper(*args, **kwargs):
        print(func.__name__)
        print(datetime.datetime.now())
    return wrapper

@func_and_now
def main_func():
    pass


main_func()
