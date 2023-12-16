# 4. Prices and sizes

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from analysis_data_1 import analysis_data

# Let's prepare the data set for estimating price correlation depending on size
analysis_data = analysis_data[["price", "depth", "height", "width"]]
analysis_data = analysis_data.dropna(subset=["depth", "height", "width"])
print(analysis_data)

# Let's calculate the correlation between price and volume of goods
analysis_data['size'] = analysis_data['width'] * analysis_data["depth"] * analysis_data["height"]
correlation = analysis_data["price"].corr(analysis_data['size'])
print(correlation)

''' In this case, the Pearson correlation coefficient between “price” and “size” is 
0.8233616402808847, indicating a strong positive correlation between the two variables. 
It means, that as the size of an object increases, the price also tends to increase, 
and vice versa. 
    The second value in the result is a p-value of 0.0. This indicates that the 
probability observing a correlation as strong as 0.8233616402808848 is only due to 
random chance actually equal to zero. Therefore, we can conclude that there is 
a significant correlation between the variables "price" and "size" in the data set.
    Overall, the result indicates a strong positive correlation between “price” and 
“size”, which means that as the size of the property increases, the price also tends 
to increase. '''

# Outliers
z_scores = zscore(analysis_data['size'])
outliers = (z_scores > 3)
print(analysis_data[outliers])

sns.scatterplot(x=analysis_data['size'], y=analysis_data['price'])
plt.title('Prices by Sizes')
plt.xlabel('')
plt.ylabel('')
plt.show()
