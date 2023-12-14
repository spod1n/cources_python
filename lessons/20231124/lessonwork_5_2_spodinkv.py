"""
Task 3. Знайти суму чисел які НЕ є голосними. Input: str, Output: int
"""

vowels = ['a', 'e', 'i', 'o', 'u']

def sum_not_vowels(_string: str) -> int:
    letters_sum = 0

    for letter in _string:
        if letter not in vowels:
            letters_sum += ord(letter)

    return letters_sum

string = input('Введіть рядок: ')

print(f'Сума приголосних літер: {sum_not_vowels(string)}')


"""
# Task 4. Перевірка просте число чи ні. Input: int, Output:str
"""

def is_prime(num: int) -> bool:
    return num > 1 and all(num % i != 0 for i in range(2, int(num**0.5) + 1))

num = int(input('Введіть ціле число: ').strip())

print(is_prime(num))


"""
Task 5. Перевірка str що це валідна IP4 - 192.128.1.1 - range [0-255].[0-255].[0-255].[0-255]
"""

def ip_valid(num: list) -> bool:
    flag_valid = []

    if len(ip_arr) == 4:
        for ip in ip_arr:
            if 0 < int(ip) <= 255:
                flag_valid.append(True)
            else:
                flag_valid.append(False)
    else:
        flag_valid.append(False)

    return all(flag_valid)

ip_arr = input('Введіть IP адрес: ').split('.')

print(ip_valid(ip_arr))
