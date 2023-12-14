# STRING
"""
Task 1. Write a program that takes a string and returns it in reverse order.
"""
string = input('Введіть рядок: ')
reverse_string = str(string)[::-1]

print(f'Реверсний рядок: {reverse_string}')


"""
Task 2. Check if the input string is a palindrome, disregarding case and spaces.
"""
string = input('Введіть рядок: ').replace(' ', '').lower()

if string == str(string)[::-1]:
    print('Ваш рядок паліндром.')

else:
    print('Рядок не є паліндромом.')


"""
Task 3. Write a program that counts the number of words in a sentence.
"""
string = input('Введіть речення: ').split(' ')

print(f'Ваше речення має {len(string)} слова.')


# LOOPS
"""
Task 1. Take as input 10 real numbers and calculate their average.
"""
arr_num = []

while len(arr_num) < 10:
    arr_num.append(float(input('Введіть число: ').strip()))

print(f'Середнє значення веденних дійсних чисел дорівнює: {sum(arr_num) / len(arr_num)}')


"""
Task 2. Print all multiples of 14 smaller than 1000.
"""
[print(num) if (num % 14) == 0 else 0 for num in range(1, 1000)]


"""
Task 3. Take an integer as input and test whether or not it is prime (a prime number is divisible only by 1 and itself).
"""
num = int(input('Введіть число: ').strip())

if num > 1 and all(num % i != 0 for i in range(2, int(num ** 0.5) + 1)):
    print(True)
else:
    print(False)


"""
Task 4. Take as input two real numbers and an operation (“+” or “-” or“/” or “*”)
and calculate the result of the operation.
"""
num_1 = int(input('Введіть перше ціле число: ').strip())
num_2 = int(input('Введіть друге ціле число: ').strip())
action_input = input('Введіть дію: ').strip()

for action in ['+', '-', '*', '/']:
    if action == action_input:
        if num_2 != 0 and action != '/':
            result = eval(f'{num_1} {action} {num_2}')
            print(f'{num_1} {action} {num_2} = {result}')
            break
        else:
            print('Ділення на нуль не можливе.')
else:
    print('Введена невідома дія.')


"""
Task *. Напишіть програму, що виводить таблицю множення, але у зворотньому порядкуЖ від 10x10 до 1x1.
"""
for num_1 in range(10, 0, -1):
    for num_2 in range(10, 0, -1):
        print(f'{num_1} x {num_2} = {num_1 * num_2} ')
