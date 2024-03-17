import pandas as pd


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class')

iris_df = pd.read_csv(URL, header=None, names=column_names)
iris_df.info()
