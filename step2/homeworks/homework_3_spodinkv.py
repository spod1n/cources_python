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

import time
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


def check_time(func):
    """ Декоратор для вимірювання часу роботи функції

    :param func: Функція яку обгортаємо
    :return: Результат обгортки
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")

        return result

    return wrapper


@check_time
def find_primes_single_thread(start: int, end: int) -> list:
    """ Функція для знаходження простих чисел в діапазоні.

    :param start: Початок діапазону перевірки
    :param end: Кінець діапазону перевірки
    :return: Прості числа
    """

    return [num for num in range(start, end + 1) if is_prime(num)]


@check_time
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
        # result_thr.extend(find_primes_single_thread(start_thr, end_thr))
        result_thr.extend([num for num in range(start_thr, end_thr + 1) if is_prime(num)])

    result1, result2 = [], []
    mid = (start + end) // 2

    thread1 = threading.Thread(target=find_primes_range, args=(start, mid, result1))
    thread2 = threading.Thread(target=find_primes_range, args=(mid + 1, end, result2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    return result1 + result2


def functions_test(start: int, end: int) -> None:
    """ Тестові випадки для обох функцій для перевірки - чи вони повертають однаковий результат.

    :param start: Початок тестового діапазону
    :param end: Кінець тестового діапазону
    :return: None
    """

    single_thread_result = find_primes_single_thread(start, end)
    multi_thread_result = find_primes_multi_thread(start, end)

    assert single_thread_result == multi_thread_result

    print('Test passed.', end='\n\n')


if __name__ == '__main__':
    # Висновок: Двупотокова перевірка числа на просте - швидше за однопотокову.
    # Різниця часу відпрацювання функцій прямопропорційна розміру діапазону.
    #

    functions_test(1, 100)

    _start, _end = 1, 1000
    print(_start, _end, sep=' - ', end=':\n')
    find_primes_single_thread(_start, _end)
    find_primes_multi_thread(_start, _end)

    _start, _end = 1, 1_000_000
    print(_start, _end, sep=' - ', end=':\n')
    find_primes_single_thread(_start, _end)
    find_primes_multi_thread(_start, _end)
