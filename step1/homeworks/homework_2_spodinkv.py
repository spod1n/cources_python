"""
Домашнє завдання #2
Климентій Сподін
Version Python 3.11.6
"""

"""
TASK #1
Як вхідні дані запитайте ціле число. Якщо воно ділиться на 3, виведіть "foo"; якщо воно ділиться на 5, виведіть "bar";
якщо воно ділиться на обидва, виведіть "ham" (а не "foo" або "bar").
"""
num = int(input('Введіть ціле число: '))

if (num % 3 == 0) and (num % 5 == 0):
    print('ham')
elif num % 3 == 0:
    print('foo')
elif num % 5 == 0:
    print('bar')
else:
    print('Число не ділиться на 3 або 5.')


"""
TASK #2
Як вхідні дані запитайте два числа та виведіть яке з них менше і яке більше.
"""
num_1 = int(input('Введіть перше ціле число: ').strip())
num_2 = int(input('Введіть друге ціле число: ').strip())

if num_1 > num_2:
    print(f'Число {num_1} більше.', f'Число {num_2} менше.', sep='\n')
elif num_1 < num_2:
    print(f'Число {num_1} менше.', f'Число {num_2} більше.', sep='\n')
else:
    print(f'Числа {num_1} та {num_2} однакові.')


"""
TASK #3
Як вхідні дані запитайте три числа і виведіть найменше, середнє та найбільше. Припустимо, всі вони різні
"""
num_1 = int(input('Введіть перше ціле число: ').strip())
num_2 = int(input('Введіть друге ціле число: ').strip())
num_3 = int(input('Введіть третє ціле число: ').strip())

nums_sort = [num_1, num_2, num_3].sort()

print(f"Найменше число: {nums_sort[0]}")
print(f"Середнє число: {nums_sort[1]}")
print(f"Найбільше число: {nums_sort[2]}")


"""
TASK #4
Зіграйте у гру Fizz-Buzz: виведіть усі числа від 1 до 100;
якщо число ділиться на 3, замість числа виведіть "fizz".
Якщо воно ділиться на 5, замість числа виведіть "Buzz".
Якщо воно ділиться на обидва, виведіть "fizz buzz" замість числа.

"""
arr_fizz_buzz = []

for num in range(1, 101):
    if (num % 3 == 0) and (num % 5 == 0):
        arr_fizz_buzz.append('fizz buzz')
    elif num % 3 == 0:
        arr_fizz_buzz.append('fizz')
    elif num % 5 == 0:
        arr_fizz_buzz.append('Buzz')
    else:
        arr_fizz_buzz.append(num)

print(arr_fizz_buzz)


"""
TASK #5
Зіграйте у гру 7-boom: виведіть усі числа від 1 до 100;
якщо число ділиться на 7 або містить цифру 7, виведіть "BOOM" замість числа.

"""
arr_7_boom = []

for num in range(1, 101):
    if (num % 7 == 0) or ('7' in str(num)):
        arr_7_boom.append('BOOM')
    else:
        arr_7_boom.append(num)

print(arr_7_boom)
