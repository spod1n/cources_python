# defaultdict -  дозволяє створювати словники із дефолт value
from collections import defaultdict, Counter
# key  - default value

# Створення defaultdict зі значенням за замовчуванням 0
my_dict = defaultdict(int)  # ключове слово

# Рядок, який потрібно проаналізувати
my_string = "AABBCA"

# Підрахунок кількості букв у рядку
for letter in my_string:
    my_dict[letter.lower()] += 1

# Виведення кількості кожної букви
for letter, count in my_dict.items():
    print(f"{letter}: {count}")

print(my_dict)
my_dict['e']
my_dict['f']
my_dict['a'] = '10'

print(my_dict)