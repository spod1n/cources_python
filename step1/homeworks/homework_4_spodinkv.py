import time


def is_prime_simple(m_num: int) -> time:
    """ Проста реалізація методу пошуку простих чисел

    :param m_num: (int) - max range
    :return: (list) - prime numbers
    """
    primes, start_time = [], time.time()

    for num in range(2, m_num + 1):
        for divisor in range(2, num):
            if num % divisor == 0:
                break
        else:
            primes.append(num)

    return time.time() - start_time


def is_prime_eratosfen(m_num: int) -> time:
    """ Ефективний метод пошуку простих чисел за методом Решета Эратосфена

    :param m_num: (int) - max range
    :return: (list) - prime numbers
    """
    start_time = time.time()
    sieve = [True] * (m_num + 1)

    for num in range(2, int(m_num ** 0.5) + 1):
        if sieve[num]:
            for multiple in range(num * num, m_num + 1, num):
                sieve[multiple] = False
    primes = [num for num in range(2, m_num + 1) if sieve[num]]

    return time.time() - start_time


# верхня межа діапазона
max_num = 10

# проста реалізація
is_simple = is_prime_simple(max_num)
print(f'Час виконання простої реалізації: {is_simple} сек.')

# метод Решето Ератосфена
is_eratosfen = is_prime_eratosfen(max_num)
print(f"Час виконання методу Решето Ератосфена: {is_eratosfen} сек.")

# висновок
if is_simple > is_eratosfen:
    print(f'Метод Решето Ератосфена ефективніший за просту реалізацію на {is_simple - is_eratosfen} сек.')
elif is_simple < is_eratosfen:
    print(f'Проста реалізація ефективніша за метод Решето Ератосфена на {is_eratosfen - is_simple} сек.')
else:
    print('Проста реалізація та метод Решето Ератосфена мають однакову ефективність.')

    # визначення верхньої межі діапазону коли методи мають не однакову ефективність
    while is_simple == is_eratosfen:
        is_simple = is_prime_simple(max_num)
        is_eratosfen = is_prime_eratosfen(max_num)

        if is_simple > is_eratosfen:
            print(f'При діапазоні [2, {max_num}] метод Решето Ератосфена ефективніший на {is_simple - is_eratosfen} сек.')
            break
        elif is_simple < is_eratosfen:
            print(f'При діапазоні [2, {max_num}] проста реалізація ефективніша на {is_eratosfen - is_simple} сек.')
            break
        else:
            max_num += 1
