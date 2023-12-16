"""
< Завдання #8 >.

Завантажте набір даних рейтингу кави та створіть візуалізації, щоб відповісти на запитання.
Ви можете використовувати додаткові типи графіків до тих, які ми дізналися на уроках, якщо ви думаєте,
що вони допоможуть відповісти на запитання.

1. Які країни є великими експортерами кави?
2. Які кореляції між різними показниками оцінки кави?
3. Який (якщо є) вплив кольору зерен на загальний сорт кави?
4. Чи впливає країна походження на якість кави?
5. Чи суттєво впливає висота на якість кави?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %%
# CONSTANTS
URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv'
SPAN_C, SPAN_L = 11, 1000   # кількість групувань та довжина прольоту

if __name__ == '__main__':
    df_coffee = pd.read_csv(URL)
    df_coffee = df_coffee.loc[df_coffee['total_cup_points'] > 0]

    # %%
    # Визначення великих експортерів кави
    country_exporters = df_coffee.groupby('country_of_origin')['number_of_bags'].sum().reset_index()
    country_exporters_top10 = country_exporters.sort_values(by='number_of_bags', ascending=False).head(5)

    # визначаю розмір фігури
    plt.figure(figsize=(10, 6))

    # налаштування фігури
    sns.barplot(data=country_exporters_top10,
                x='country_of_origin', y='number_of_bags',
                palette='viridis', hue='country_of_origin',
                legend=False)

    # назви осей та фігури
    plt.title('Top Coffee Exporting Countries')
    plt.xlabel('Country')
    plt.ylabel('Number Of Bags')

    plt.tight_layout()  # підганяю розмір вікна до розміру фігури
    plt.show()

    # %%
    # Кореляції між різними показниками оцінки кави
    numeric_columns = df_coffee.select_dtypes(include=['float64', 'int64']).columns
    numeric_data = df_coffee[numeric_columns]

    # обчислюю кореляційну матрицю
    matrix_corr = numeric_data.corr()

    # генерую маску верхнього трикутника
    mask = np.triu(np.ones_like(matrix_corr, dtype=bool))

    f, ax = plt.subplots(figsize=(15, 8))                   # розміри фігури
    cmap = sns.diverging_palette(230, 20, as_cmap=True)     # палітра

    # налаштовую теплову карту
    sns.heatmap(matrix_corr, mask=mask, cmap=cmap, annot=True)

    # назва фігури
    plt.title('Top Coffee Exporting Countries')

    plt.tight_layout()  # підганяю розмір вікна до розміру фігури
    plt.show()

    # %%
    # Вплив кольору зерен на загальний сорт кави
    plt.figure(figsize=(12, 6))

    sns.boxplot(data=df_coffee,
                x='color', y='total_cup_points',
                palette='viridis', hue='color',
                legend=False)

    # назви осей
    plt.title('Impact of Bean Color on Coffee Ratings')
    plt.xlabel('Bean Color')
    plt.ylabel('Total Cup Points')

    plt.tight_layout()  # підганяю розмір вікна до розміру фігури
    plt.show()

    # %%
    # Вплив країни походження на якість кави
    plt.figure(figsize=(14, 8))     # розміри фігури

    # налаштування фігури
    sns.boxplot(data=df_coffee,
                x='country_of_origin', y='total_cup_points',
                palette='viridis', hue='country_of_origin',
                legend=False)

    # назви осей та фігури
    plt.title('Impact of Origin Country on Coffee Ratings')
    plt.xlabel('Country of Origin')
    plt.xticks(rotation=45, ha='right')     # наклон підписів осі x

    plt.ylabel('Total Cup Points')

    plt.tight_layout()  # підганяю розмір вікна до розміру фігури
    plt.show()

    # %%
    # Вплив висоти на якість кави
    bins = [-float('inf')] + [(i + 1) * SPAN_L - 1 for i in range(SPAN_C - 1)] + [float('inf')]
    labels = [f'{i * SPAN_L}-{(i + 1) * SPAN_L - 1}' for i in range(SPAN_C - 1)] + [f'>{(SPAN_C - 1) * SPAN_L}']

    # перетворюю фути на метри
    df_coffee['altitude_mean_meters'] = df_coffee.apply(
        lambda r: r['altitude_mean_meters'] * 0.3048 if r['unit_of_measurement'] == 'ft' else r['altitude_mean_meters'],
        axis=1)

    # групую середню висоту в метрах
    df_coffee['altitude_group'] = pd.cut(df_coffee['altitude_mean_meters'], bins=bins, labels=labels, right=False)

    plt.figure(figsize=(10, 6))     # розміри фігури

    # налаштування фігури
    sns.scatterplot(data=df_coffee, x='total_cup_points', y='altitude_group', color='green')

    # назви осей та фігури
    plt.title('Impact of AVG Altitude on Coffee Ratings')
    plt.xlabel('Total Cup Points')
    plt.ylabel('AVG Altitude Group')

    plt.tight_layout()  # підганяю розмір вікна до розміру фігури
    plt.show()
