# Lambda function and reduce/filter/map 
# Task 6. Використайте reduce() для обчислення суми всіх елементів у списку: [1, 2, 3, 4, 5] → 15 (1 + 2 + 3 + 4 + 5).
# Task 7. Використайте map() для перетворення списку зі рядок на список їхніх довжин: ['apple', 'banana', 'cherry'] → [5, 6, 6].
# Task 8. Використайте filter() для знаходження всіх елементів списку, які більше за середнє значення списку: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] → [6, 7, 8, 9, 10].
# Task 9. Використайте reduce() для обчислення найбільшого елемента у списку: [1, 5, 3, 9, 2] → 9.
# Task 10. Використайте map() для перетворення списку зі рядок на список великих літер: ['apple', 'banana', 'cherry'] → ['APPLE', 'BANANA', 'CHERRY'].

from functools import reduce 
# Task 6
arr = [i for i in range(1, 100)]

# first var
result = reduce(lambda x, y: x + y, arr)
print(result)

# second var
result = sum(i for i in arr if isinstance(i, (int, float)))
print(result) 

# thierd var
result = sum(arr)
print(result)

# 4-var
count_ = 0
for i in arr:
    count_ += i 
print(count_)


# Task 7 

# var 1 
arr = ['apple', 'banana', 'cherry']

result = list(map(lambda x: (x,len(x)), arr))

print(arr, result)


# Task 8 

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 1 var 
avg_ = sum(arr) / len(arr)

result = list(filter(lambda x: x > avg_, arr))
print(avg_, result)

# 2 var 
from statistics import mean 

avg_ = mean(arr) 
result = list(filter(lambda x: x > avg_, arr))
print(avg_, result)



# Task 9 

# var 1
arr = [1, 5, 3, 9, 2]
print(max(arr), arr)

# var 2
print(sorted(arr)[-1], arr)

# var 3 
result = reduce(lambda x, y: max(x, y), arr)
print(result, arr)


# Task 10 
arr = ['apple', 'banana', 'cherry']

result = list(map(lambda x: x.upper(), arr))
print(arr, result)


# Task: try/except
# 1. Написати калькультор, який при діленні на нуль повертає текст помилки, опрацювати інші можливі помилки також 
# 2. Написати функцію, яка при вводі юзера не числа, буде повторювати дію, до тих пір поки user - не введе позитивне число 



# Read file tasks
# 1. Створити файл та записати в нього словних: {'UK': 'London', 'Germany': 'Berlin', 'Ukraine': 'Kyiv'}
# Формат запису: Country: name_of_contry. Capital is: name_of_capital

# 2. Прочитати файл Japan, та в нього добавити інформацію про державний устрій(Кон монархія). Зробити це за допомогою context manager

# 3. Прочитати файл Japan та записати його зміст в інший файл Japan_Backup в назву файлу добавити поточну дату, зробити це за допомогою context_manager

# 4. Створити файли де назвою буде алф від A-Z, записати в них будь-яке число(бібліотека random), зробити за допомогою context_manager

# Task 1
capitals = {'UK': 'London', 'Germany': 'Berlin', 'Ukraine': 'Kyiv'}
 
with open('capitals.txt', 'w') as file:
    for country, capital in capitals.items():
        file.writelines(f'Country: {country}. Capital is: {capital}\n')

# Task 2

with open('Japan.txt', 'a') as file:
    file.write('\nустрій(Кон монархія)\n')

# Task 3
import datetime 

now = datetime.datetime.today()

with open('Japan.txt', 'r') as file, open(f'JapanBackup{now}.txt', 'w') as back_file:
    for i in file.readlines():
        back_file.writelines(i)

# Task 4
import random as rn    

def generate_value(n: int = 1, m: int = 100) -> int:
    return str(rn.randint(n, m)) if m > n else '1'


def generate_alp(start_number: int = 65, end_number: int = 90) -> str:
    return ''.join(chr(i) for i in range(start_number, end_number + 1 )) if end_number > start_number else None 


for item in generate_alp():
    with open(f'{item}.txt', 'w') as file:
        file.write(generate_value())

