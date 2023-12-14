import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore, pearsonr

from analysis_data_1 import analysis_data

analysis_data = analysis_data[["price", "depth", "height", "width"]]
analysis_data = analysis_data.dropna(subset=["depth", "height", "width"])
print(analysis_data)

# Let's calculate the pairwise correlation between numerical columns
correlation_matrix = analysis_data.corr()
print(correlation_matrix)

# Or
print(pearsonr(analysis_data["price"], analysis_data["width"]))
print(pearsonr(analysis_data["price"], analysis_data["depth"]))
print(pearsonr(analysis_data["price"], analysis_data["height"]))

''' There is a strong positive correlation between “price” and “width” (correlation 
coefficient 0.768), which suggests that as the width of the object increases, the price 
also tends to increase. The p value is very small (2.93e-303), indicating that this 
correlation is statistically significant.
    There is a moderate positive correlation between “price” and “depth” (correlation 
coefficient 0.635), which suggests that as the depth of the object increases, the price 
tends to increase. The p value is very small (3.24e-176), which suggests that this 
correlation is also statistically significant.
    There is a weak positive correlation between “price” and “height” (correlation 
coefficient 0.228), which indicates a slight upward trend in price as the height of 
the property increases. The p value is small (9.59e-20), which suggests that this 
correlation is statistically significant.
    So, "width" and "depth" are important factors in determining the price of an object, 
while “height” has a weaker relationship with price.
    Overall, these results indicate that there is a strong positive linear relationship 
between price and depth and height, as well as a weaker, but still statistically 
significant positive linear relationship between price and width. '''

# Outliers
z_scores = zscore(analysis_data)
abs_z_scores = np.abs(z_scores)
outlier_indices = np.where(abs_z_scores > 3)[0]
outliers = analysis_data.iloc[outlier_indices]
print(outliers)

# Heatmaps
correlation_matrix.index = correlation_matrix.index.map(str)
correlation_matrix.columns = correlation_matrix.columns.map(str)

fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix, cmap='coolwarm')

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(['price', 'depth', 'height', 'width'])))
ax.set_yticks(np.arange(len(['price', 'depth', 'height', 'width'])))
ax.set_xticklabels(['price', 'depth', 'height', 'width'], rotation=45, ha="right")
ax.set_yticklabels(['price', 'depth', 'height', 'width'])

for i in range(len(['price', 'depth', 'height', 'width'])):
    for j in range(len(['price', 'depth', 'height', 'width'])):
        text = ax.text(j, i, '{:.2f}'.format(correlation_matrix.iloc[i, j]), ha="center", va="center",
                       color="w")
plt.title("Correlation matrix between Price and Sizes")
plt.show()
