"""
# Task 2. Написати декоратор, де буде перед викликом кожної функції delay 30 seconds
"""


import time


def delay_front(func):
    def wrapper(*args, **kwargs):
        time.sleep(5)
        func(*args, **kwargs)
    return wrapper


@delay_front
def main_func():
    print('Main function start')


main_func()
