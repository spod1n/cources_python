""" Klymnetii Spodin. Homework Scikit-Learn """

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# %%
# Форма та тип даних. Перші 3 рядки
iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

print('Форма та тип даних:')
iris_df.info()
print('', 'Перші 3 рядки:', iris_df.head(3).to_markdown(), sep='\n', end='\n\n')

# %%
# Ключі, кількість рядків-стовпців, назви ознак та опис даних Ірису
print('Ключі:', iris_data.keys(), sep='\n', end='\n\n')
print('Кількість рядків та стовпців:', iris_data.data.shape, sep='\n', end='\n\n')
print('Назви ознак:', iris_data.feature_names, sep='\n', end='\n\n')
print('Опис даних Ірису:', iris_data.DESCR, sep='\n', end='\n\n')

# %%
# Базові статистичні деталі даних Ірису
print('Базові статистичні деталі:', iris_df.describe(), sep='\n', end='\n\n')

# %%
# Отримання спостережень кожного виду
iris_types = iris_data.target_names

for i, iris_type in enumerate(iris_types):
    iris_types_df = iris_df[iris_df['target'] == i]
    print(f'Спостереження для виду {iris_type}: {iris_types_df.shape[0]}',
          iris_df[iris_df['target'] == i].reset_index(drop=True).to_markdown(), sep='\n', end='\n\n')

# %%
# Створення графіку для отримання загальної статистики даних Ірис
sns.pairplot(iris_df, hue='target')
plt.show()


# %%
# Стовпчаста діаграма для визначення частоти трьох видів Ірису
iris_types_df = iris_df.copy()
iris_types_df['species'] = iris_data.target_names[iris_types_df.target]
sns.countplot(data=iris_types_df, x='species', hue='species')
plt.xlabel('Види Ірису')
plt.ylabel('Частота')
plt.title('Частота видів Ірису')

plt.tight_layout()
plt.show()

# %%
# Розділення набору даних на X (атрибути) та y (мітки)
X = iris_df.drop('target', axis=1)
y = iris_df['target']

# %%
# Розділення набору даних на тренувальні та тестові дані (70% на 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Перетворення стовпців видів на числовий стовпець
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# %%
# Виведення розмірів тренувальних та тестових наборів
print('Розмір тренувального набору (70%):', X_train.shape, y_train_encoded.shape, sep=' ', end='\n')
print('Розмір тестового набору (30%):', X_test.shape, y_test_encoded.shape, sep=' ', end='\n\n')

# %%
# Прогнозування відповіді за допомогою алгоритму найближчих сусідів (K Nearest Neighbor Algorithm)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train_encoded)
y_pred = knn.predict(X_test)

# %%
# Виведення прогнозованих відповідей для тестового набору даних
print('Прогнозовані відповіді для тестового набору:', label_encoder.inverse_transform(y_pred), sep='\n', end='\n\n')

# %%
# Оцінка R2 для моделі KNeighborsClassifier
r2 = r2_score(y_test, y_pred)
print("R2 для моделі KNeighborsClassifier:", r2)
