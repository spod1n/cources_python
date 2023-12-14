"""
Task 0. Перевірка на парність числа. Input: int, Output: bool
"""
def is_even(_num: int) -> bool:
    if _num % 2 == 0:
        return True
    else:
        return False

num = int(input('Введіть число: ').strip())

print(is_even(num))


"""
# Task 1. Повернути максимальне число з inf параметрів. Input: int, int, int, Output: max - number
"""
arr_num = []

def max_num(*args) -> int:
    return max(*args)

for i in range(0, 3):
    num = int(input('Введіть число: ').strip())
    arr_num.append(num)

print(f'Максимальне введене число: {max_num(arr_num)}')


"""
Task 2. Перевірка на паліндром. Input: str Output: bool
"""

def is_palindrome(string: str) -> bool:
    if string == str(string)[::-1]:
        return True
    else:
        return False

string = input('Введіть рядок: ').replace(' ', '').lower()

print(is_palindrome(string))