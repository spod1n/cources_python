pip install pandas
pip install seaborn
pip install matplotlib


https://colab.research.google.com/?hl=ru


import warnings
warnings.simplefil  ter(action='ignore', category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%matplotlib inline

sns.get_dataset_names()

df = sns.load_dataset('iris')
df.head() # показати n - перших записів, n - default 5
df.tail()
df.sample(n = 5)
df.info()
df.select_dtypes(include=['float64'])
df.describe()
df.isnull()
df.isnull().sum()
df['species']
df['species'].values
df['species'].value_counts()
df['species'].value_counts(normalize= True)
df['species'].hist();
df['species'].unique()
df['species'].nunique()
df['sepal_length'] > 5.0
df[df['sepal_length'] > 5.0]
df[(df['sepal_length'] >= 3.15) & (df['petal_length'] <= 4.5)]

# Upload tips(seaborn) - df_tips
# EDA(head, tail, sample, hist, info, shape, unique, ...)

