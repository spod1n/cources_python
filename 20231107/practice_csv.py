"""
# Task 3. Read csv file, calculate sum of columns(int), display information about csv
"""

import csv
import re

file = 'mcdonalds_dataset.csv'
product_price = 0
protein = 0

with open(file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file, delimiter=',')

    for field in csv_reader:
        product_price += float(re.sub(r'[^0-9.]', '', field['product_price']))

    print(f'Загальна ціна: {round(product_price, 2)}')
