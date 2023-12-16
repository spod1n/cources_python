"""
# Task 3. Read csv file, calculate sum of columns(int), display information about csv
"""

import csv

file = 'mcdonalds_dataset.csv'
product_price = 0
protein = 0

with open(file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file, delimiter=',')

    for field in csv_reader:
        if '£' in field['product_price']:
            product_price += float(field['product_price'].replace('£', '').strip())
        elif 'P' in field['product_price']:
            product_price += float(f"0.{field['product_price'].replace('P', '').strip()}")

    print(f'Загальна ціна: {round(product_price, 2)} £')
