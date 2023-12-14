"""
Task 1. Напишіть функцію, яка приймає два списки і повертає список спільних елементів.
Використовуйте множини для знаходження спільних елементів
"""
arr1 = [1, 2, 3, 4]
arr2 = [3, 4, 5, 6]


def intersection_set(arr1:list, arr2:list) -> set:
    return set(arr1).intersection(set(arr2))

print(intersection_set(arr1, arr2))


"""
# Task 2. Напишіть функцію, яка приймає список та повертає True, якщо усі елементи унікальні,
та False, якщо є дублікати. Використовуйте множини для визначення унікальних елементів.
"""
arr = [1, 2, 3, 4, 4]


def unique_elements(arr:list) -> bool:
    return len(arr) == len(set(arr))

print(unique_elements(arr))


"""
# Task 3. Напишіть функцію, яка приймає список і повертає список без дублікатів.
Використовуйте множини для видалення дублікатів
"""
arr = [1, 2, 3, 4, 4]


def remove_duplicates(arr:list) -> list:
    return list(set(arr))

print(remove_duplicates(arr))
