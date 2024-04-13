""" Klymnetii Spodin. Homework seaborn """

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_penguins = sns.load_dataset('penguins')
print(df_penguins.head().to_markdown())
sns.set(style='whitegrid')

# %%
# Аналіз розподілу ваги та висоти птахів:
"""
Спробуйте побудувати графік, який відображає розподіл ваги та висоти птахів (пінгвінів) за допомогою Seaborn.
Використовуйте відповідні графічні методи для визначення розподілу та залежності між цими двома ознаками.
"""
# Висоти птахів (пінгвінів) немає у датасеті. Використав довжину лапи.
plt.figure(figsize=(10, 7))

sns.scatterplot(data=df_penguins, x='body_mass_g', y='flipper_length_mm', hue='species')

plt.title('Розподіл ваги та довжини лапи птахів (пінгвінів)')
plt.xlabel('Вага, г')
plt.ylabel('Довжина лапи, мм')
plt.legend(title='Вид')

plt.tight_layout()
plt.show()

# Чим більша вага - тим більше довжина лапи.

# %%
# Вивчення впливу виду птаха на розміри крил:
"""
Зобразіть графік, який показує розміри крил у залежності від виду птаха.
Використовуйте різні кольори або стилі для різних видів птахів.
Це дозволяє вам визначити, чи існують різниці у розмірах крил між різними видами пінгвінів.

"""
plt.figure(figsize=(7, 7))

sns.swarmplot(data=df_penguins, x='sex', y='flipper_length_mm', hue='species')

plt.title('Розміри крил у залежності від виду та статі птаха')
plt.xlabel('Стать птаха')
plt.ylabel('Довжина лапи, мм')
plt.legend(title='Вид птаха')

plt.tight_layout()
plt.show()

# Adelie та Chinstrap мають однаковий розподіл розмірів довжини лапи. Gentoo - значно переважає в розмірах лап.
# Жіноча стать має менші по довжині лапи, ніж чоловіча стать.

# %%
# Кореляція між різними ознаками:
"""
Спробуйте побудувати теплову карту кореляції для всього набору даних пінгвінів.
Вивчення кореляцій може допомогти зрозуміти, які ознаки співвідносяться між собою.
Зробіть аналіз кореляцій та виберіть ті ознаки, які мають високий ступінь взаємозв'язку.
"""
numeric_columns = df_penguins.select_dtypes(include=['float64', 'int64']).columns
numeric_data = df_penguins[numeric_columns]

matrix_corr = numeric_data.corr()
mask = np.triu(np.ones_like(matrix_corr, dtype=bool))

f, ax = plt.subplots(figsize=(10, 3))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(matrix_corr, mask=mask, cmap=cmap, annot=True)

plt.title('Теплова карта кореляції для набору числових даних пінгвінів')

plt.tight_layout()
plt.show()

# Довжина дзьоба (bill_length_mm) має позитивну кореляцію з довжиною лапок (flipper_length_mm)
# і масою тіла (body_mass_g). Це логічно, оскільки більші пінгвіни зазвичай мають більші дзьоби,
# лапки та масу тіла.
#
# Довжина лапок (flipper_length_mm) також корелює з масою тіла (body_mass_g), що також логічно,
# оскільки більші пінгвіни мають більші лапки і зазвичай важчі.
#
# Негативна кореляція спостерігається між довжиною дзьоба (bill_length_mm) і глибиною дзьоба (bill_depth_mm),
# що може бути пов'язано з різноманітністю видів пінгвінів.
