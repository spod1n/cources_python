"""
Давайте розглянемо завдання, де маємо визначити ймовірність отримання певної комбінації
карт у грі зі звичайною колодою карт (52 карти).

Для цього використаємо модуль itertools, зокрема функцію combinations.
"""

from itertools import combinations


def calculate_probability(combination):
    total_combinations = len(list(combinations(range(TOTAL_CARDS), len(combination))))

    target = 0
    for combination_num in combinations(range(TOTAL_CARDS), len(combination)):
        if set(combination) == set(combination_num):
            target += 1

    return target / total_combinations


# test
TOTAL_CARDS = 52
test1 = [0, 1, 2, 3, 4]

result = calculate_probability(test1)
print(f'Ймовірність отримання комбінації {test1}: {result}')
