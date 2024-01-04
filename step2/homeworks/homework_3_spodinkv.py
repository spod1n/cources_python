"""
Назва завдання: Пошук простих чисел
Опис: Створіть програму, яка шукає всі прості числа в заданому діапазоні. Ви реалізуєте два варіанти пошуку - один
використовуючи один потік, а інший використовуючи багатопоточність.

Кроки:
1. Напишіть функцію is_prime(n), яка приймає число n та повертає True, якщо воно є простим, та False в іншому випадку.

2. Напишіть функцію find_primes_single_thread(start, end), яка знаходить всі прості числа у діапазоні від start до end
за допомогою одного потоку.

3. Напишіть функцію find_primes_multi_thread(start, end), яка робить те ж саме, але використовуючи багатопоточність.
Розділіть діапазон на дві частини та обчисліть прості числа паралельно вдвох потоках. Потім об'єднайте результати.

4. Створіть тестові випадки для обох функцій та перевірте, чи вони повертають однаковий результат.

5. Виміряйте час виконання кожної з функцій для різних діапазонів чисел та порівняйте їх ефективність.

6. Зробіть аналіз результатів та поясніть, чому один варіант може бути ефективніший за інший в певних умовах.

7. Закінчіть завдання, надславши код програми в ЛМС
"""

import threading


def is_prime(num: int) -> bool:
    """ Функція для перевірки числа на просте.

    :param num: Вхідне число
    :return: Просте число чи ні
    """

    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True


def find_primes_single_thread(start: int, end: int) -> list:
    """ Функція для знаходження простих чисел в діапазоні.

    :param start: Початок діапазону перевірки
    :param end: Кінець діапазону перевірки
    :return: Прості числа
    """

    return [num for num in range(start, end + 1) if is_prime(num)]


def find_primes_multi_thread(start, end):
    """ Функція для знаходження простих чисел в діапазоні у двох потоках.

    :param start: Початок діапазону перевірки
    :param end: Кінець діапазону перевірки
    :return: Прості числа
    """

    def find_primes_range(start_thr, end_thr, result_thr):
        """ Функція для потоку. Додає до результату потоку дані розширенням списку.

        :param start_thr: Початок діапазону перевірки потоку
        :param end_thr: Кінець діапазону перевірки потоку
        :param result_thr: Список з результатами потоку
        :return: Прості числа
        """
        result_thr.extend(find_primes_single_thread(start_thr, end_thr))

    result1, result2 = [], []
    mid = (start + end) // 2

    thread1 = threading.Thread(target=find_primes_range, args=(start, mid, result1))
    thread2 = threading.Thread(target=find_primes_range, args=(mid+1, end, result2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    return result1 + result2


print(find_primes_multi_thread(-100, 100))
