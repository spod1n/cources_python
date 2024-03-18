import pandas as pd


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class')

iris_df = pd.read_csv(URL, header=None, names=column_names)
iris_df.info()
print()

# 1.1 Яка середня довжина чашелистика (sepal length) для кожного виду ірису?
mean_sepal_length = iris_df.groupby('class')['sepal_length'].mean()
print('sepal length:', mean_sepal_length.to_markdown(), sep='\n', end='\n\n')

# 1.2 Яка максимальна ширина листка (petal width) для виду “setosa”?
max_petal_width_setosa = iris_df[iris_df['class'] == 'Iris-setosa']['petal_width'].max()
print(f'{max_petal_width_setosa=}', end='\n\n')

# 1.3 Яка розподіленість довжини листка (petal length) для всіх ірисів?
petal_length_distribution = iris_df['petal_length'].describe()
print('describe petal length:', petal_length_distribution.to_markdown(), sep='\n', end='\n\n')


# 2.1 Створіть новий DataFrame, в якому будуть лише дані для ірисів виду “versicolor”.
versicolor_df = iris_df[iris_df['class'] == 'Iris-versicolor'].copy()
print('iris-versicolor only:', versicolor_df.to_markdown(), sep='\n', end='\n\n')

# 2.2 Відфільтруйте дані для ірисів з довжиною листка (petal length) більше 5.0.
petal_length_df = iris_df.loc[iris_df['petal_length'] > 5.0].sort_values(by='petal_length')
print('petal length more than 5:', petal_length_df.to_markdown(), sep='\n', end='\n\n')


# 3.1 Яка середня ширина листка (petal width) для кожного виду ірису?
# 3.2 Яка мінімальна довжина чашелистика (sepal length) для кожного виду ірису?
# 3.3 Скільки ірисів кожного виду мають довжину листка (petal length) більше за середню довжину листка всіх ірисів?
