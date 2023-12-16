from itertools import combinations
from collections import Counter


FILE_NAME = 'orders.txt'
# test = [['A', 'B', 'C', 'A', 'D'], ['A', 'F'], ['B', 'C', 'A'], ['B', 'C', 'D', 'A', 'D']]


def read_file(file_name_str: str, row_spliter: str = '\n\n', val_spliter: str = '@@@') -> list[list]:
    """ Функція для вичитки файлу. Повертає словник словників.

    :param file_name_str: (str) шлях до файлу
    :param row_spliter: (str) роздільник рядків
    :param val_spliter: (str) роздільник для значень
    :return: list(list() ... list())
    """
    with open(file_name_str, 'r') as file:
        content = file.read()
        rows = content.split(row_spliter)
        values = [value.strip().split(val_spliter) for value in rows]
    return values


def quantity_count(purchases: list[list]) -> list[dict, dict]:
    """ Функція яка повертає кількість продаж в розрізі всі продуктів та їх унікальних пар.
    В першому словнику в ключах назви продуктів, в значеннях - кількість.
    В другому словнику унікальні пари продуктів, в значеннях - кількість.

    :param purchases: (list[list]) продажі по замовленням
    :return: list(dict(), dict())
    """
    product_count, pair_count = Counter(), Counter()

    for order in purchases:
        order_unique = set(order)

        # створюю словник з продуктами і їх кількістю
        for product in order_unique:
            product_count[product] += 1

        # створюю словник з парами продуктів в розрізі кожного замовлення
        # сортування значення ітерації дозволяє в один ключ словника записати реверсні пари продуктів
        for pair in combinations(order_unique, 2):
            pair_count[tuple(sorted(pair))] += 1

    return [dict(product_count), dict(pair_count)]


def confidence_and_support(product_count: dict, pair_count: dict, order_count: int) -> None:
    """ Функція повертає підтримку та впевненість для кожної пари продуктів.

    :param product_count: (dict) кількість продаж в розрізі продукту
    :param pair_count: (dict) кількість продаж в розрізі унікальних пар продуктів в рамках замовлення
    :param order_count: (int) загальна кількість замовлень
    :return: None
    """
    # обмежую підрахунок та виведення підтримки та впевненості від 0.15% та 0.45% відповідно
    min_support, min_confidence = 0.15, 0.45

    for (product_a, product_b), count in pair_count.items():
        # підтримку рахую один раз для пари продуктів, тому що вона симетрична
        support_ab = count / order_count * 100

        if support_ab >= min_support:
            # впевненість рахую для кожної пари реверсно
            confidence_a = count / product_count[product_a] * 100
            confidence_b = count / product_count[product_b] * 100

            if confidence_a >= min_confidence:
                print(f'{product_a} => {product_b}: '
                      f'(Support: {round(support_ab, 2)}%, Confidence: {round(confidence_a, 2)}%)')

            if confidence_b >= min_confidence:
                print(f'{product_b} => {product_a}: '
                      f'(Support: {round(support_ab, 2)}%, Confidence: {round(confidence_b, 2)}%)')


file_content = read_file(FILE_NAME)
products, pairs = quantity_count(file_content)
confidence_and_support(products, pairs, len(file_content))
