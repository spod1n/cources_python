import seaborn as sns
import pandas as pd

# %%
# DATASET
file_name = 'penguins.csv'
penguins = sns.load_dataset("penguins")
penguins.to_csv(file_name, index=False)
del penguins


# %%
# Аналіз даних про пінгвінів:
# 1. Зчитайте дані з CSV-файлу про пінгвінів та виведіть перші та останні 5 рядків, щоб отримати уявлення про зміст даних.
df_penguins = pd.read_csv(file_name)
print('Перші 5 рядків:\n', df_penguins.head().to_markdown(), end='\n\n')
print('Останні 5 рядків:\n', df_penguins.tail().to_markdown(), end='\n\n')

# 2. Виведіть загальну інформацію про DataFrame, включаючи типи даних та виявлення пропущених значень.
print('Інформація про датафрейм:')
df_penguins.info()
print('\n')

# 3. Обчисліть основні статистичні характеристики для числових стовпців датасету, такі як середнє значення, медіана, максимум та мінімум.
print('Статистичні характеристики для числових стовпців датасету:\n', df_penguins.describe().to_markdown(), end='\n\n')


# %%
# Вибірка та фільтрація:
# 1. Виберіть лише ті рядки, де значення у певному стовпці задовольняють певній умові
df_chinstrap = df_penguins.loc[df_penguins['species'].str.lower() == 'chinstrap']
print("Рядки де поле species 'species' дорівнює 'Chinstrap' (без урахування регістру):\n", df_chinstrap.to_markdown(), end='\n\n')

# 2. Виберіть певні стовпці для подальшого аналізу
df_some_columns = df_penguins[['species', 'island', 'bill_length_mm', 'bill_depth_mm']]
print('Тільки деякі поля:\n', df_some_columns.to_markdown(), end='\n\n')


# 3. Застосуйте фільтрацію для вибору тільки рядків з пропущеними значеннями. Все з прикладом на наборі даних пінгвіни
rows_miss_val = df_penguins[df_penguins.isnull().any(axis=1)]
print('Рядків в яких є хоча б одне нульове значення (nan):\n', rows_miss_val.to_markdown(), end='\n\n')


# %%
# Групування та агрегація:
# 1. Згрупуйте дані за певним стовпцем та обчисліть агреговані статистики для інших стовпців
df_penguins_gr = df_penguins.groupby('species')[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].mean()
print('Згруповані дані:\n', df_penguins_gr.to_markdown(), end='\n\n')

# 2. Обчисліть кількість унікальних значень у певному стовпці
uniq_val_cnt = df_penguins['island'].nunique()
print("Кількість унікальних значень у стовпці 'island': ", uniq_val_cnt)

# 3. Застосуйте власну функцію агрегації до стовпців (наприклад, функція, яка знаходить різницю між максимальним та мінімальним віком)
delta_max_min_flipper = df_penguins['flipper_length_mm'].agg(lambda col: col.max() - col.min())
print('Різниця між максимальною та мінімальною довжиною ласти:', delta_max_min_flipper)
