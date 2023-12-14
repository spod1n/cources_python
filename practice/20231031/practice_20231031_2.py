'''
# 1. Створити файл та записати в нього словних: {'UK': 'London', 'Germany': 'Berlin', 'Ukraine': 'Kyiv'}
Формат запису: Country: name_of_contry. Capital is: name_of_capital
'''
ccs = {'UK': 'London', 'Germany': 'Berlin', 'Ukraine': 'Kyiv'}

with open('countries_and_capitals.txt', 'w') as file:
    for cc in ccs.items():
        file.writelines(f'Country: {cc[0]}. Capital is: {cc[1]}\n')


'''# 2. Прочитати файл Japan, та в нього добавити інформацію про державний устрій(Кон монархія). Зробити це за допомогою context manager'''
with open('Japan.txt', 'r+') as file:
    print(file.read())
    file.write('\nThe state system: Con monarchy')


'''# 3. Прочитати файл Japan та записати його зміст в інший файл Japan_Backup, зробити це за допомогою context_manager'''
from datetime import datetime
import os

today_dt = datetime.today().strftime('%Y-%m-%d')
os.makedirs('.\Japan') if not os.path.exists('.\Japan') else False

with open('Japan.txt', 'r') as file1, open(f'.\Japan\Japan_backup_{today_dt}.txt', 'w') as file2:
    data = file1.read()
    file2.write(data)

'''# 4. Створити файли де назвою буде алф від A-Z, записати в них будь-яке число(бібліотека random), зробити за допомогою context_manager'''
import random
import string
import os

os.makedirs('.\FilesGenerator') if not os.path.exists('.\FilesGenerator') else False
eng_sting = string.ascii_uppercase

for i in range(0, 26):
    with open(f'.\FilesGenerator\{eng_sting[i]}.txt', 'w') as file:
        random_int = random.randint(1, 150)
        file.write(str(random_int))
else:
    print('Файли створені.')