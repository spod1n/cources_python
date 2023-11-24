''' 2. Perform exploratory analysis on a dataset including
descriptive statistics and visualizations. Describe the results. '''

import matplotlib.pyplot as plt

from analysis_data_1 import analysis_data

# Dataset information
print(analysis_data.shape)
print(analysis_data.info())

# Descriptive statistics
print(analysis_data.describe())

# Descriptive statistics by price
print(analysis_data["price"].describe())

plt.hist(analysis_data["price"], bins=10)
plt.title("Price statistics")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.axvline(analysis_data["price"].mean(), color='b', linestyle='dashed', linewidth=1,
            label="mean: {:.1f}".format(analysis_data["price"].mean()))
plt.axvline(analysis_data["price"].median(), color='y', linestyle='dashed', linewidth=1,
            label="median: {:.1f}".format(analysis_data["price"].median()))
plt.axvline(analysis_data["price"].max(), color='r', linestyle='dashed', linewidth=1,
            label="max: {:.1f}".format(analysis_data["price"].max()))
plt.axvline(analysis_data["price"].min(), color='black', linestyle='dashed', linewidth=1,
            label="min: {:.1f}".format(analysis_data["price"].min()))

plt.text(analysis_data["price"].mean(), 2020, "mean", rotation=90, va="baseline")
plt.text(analysis_data["price"].median(), 2020, "median", rotation=90, va="baseline")
plt.text(analysis_data["price"].max(), 2020, "max", rotation=90, va="baseline")
plt.text(analysis_data["price"].min(), 2020, "min", rotation=90, va="baseline")

plt.legend()
plt.show()
