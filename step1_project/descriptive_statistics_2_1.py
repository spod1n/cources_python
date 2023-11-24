# 1. Цены и производители

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

from analysis_data_1 import analysis_data

# Product information: names of manufacturers and number of items by name
print(analysis_data["name"].unique())
value_counts_by_name = analysis_data.groupby("name")["price"].agg(["count"]).reset_index()
value_counts_by_name = value_counts_by_name[value_counts_by_name['count'] >= 30]\
    .sort_values('count', ascending=False).reset_index(drop=True)
print(value_counts_by_name)

plt.figure(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
plt.bar(value_counts_by_name['name'], value_counts_by_name['count'],
        color=gradient_colors(value_counts_by_name['count'] / value_counts_by_name['count'].max()))

plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(value_counts_by_name['count']):
    plt.text(i, v + 2.5, str(v), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.01'))

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Number of titles by Manufacturers")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y', linestyle='--')
plt.ylim(10, value_counts_by_name['count'].max() * 1.1)
plt.subplots_adjust(bottom=0.15, left=0.1)

plt.show() 

# Descriptive statistics for price by manufacturer name
print(analysis_data.groupby("name")['price'].describe())

# Median price depending on manufacturer name
median_price_name = analysis_data.groupby("name")["price"].median().reset_index()
median_price_name = median_price_name.sort_values('price', ascending=False).reset_index(drop=True)
median_price_name = median_price_name.iloc[:21]
print(median_price_name)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FF7F00", "#FFFF00", "#00FF00"])
sns.barplot(x="price", y="name", data=median_price_name,
            palette=cmap(median_price_name['price']/median_price_name['price'].max()),
            orient='horizontal')

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax)

ax.set_xlim(150, median_price_name['price'].max() * 1.1)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.title("Median price by Manufacturers")
plt.ylabel("")
plt.xlabel("")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Gradient in the background
gradient_cmap = mcolors.LinearSegmentedColormap.from_list("", ["#002FFF", "#FF0000"])
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
im = ax.imshow(gradient, cmap=gradient_cmap, aspect='auto', extent=[150, median_price_name['price'].max() * 1.1,
                                                                    -0.5, len(median_price_name)-0.5])

for i, v in enumerate(median_price_name['price']):
    ax.text(v + 400, i, str(round(v)), ha='center', va='center', color="black", fontsize=11,
            bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.05'))

plt.subplots_adjust(left=0.32, bottom=0.1)
plt.gca().invert_yaxis()
plt.show()

formula = "price ~ name"
model = smf.ols(formula=formula, data=analysis_data).fit()
print(model.summary())

''' Result of multiple linear regression model using conventional method Ordinary Least Squares (OLS).
    The dependent variable is “price” and the model includes 606 independent variables.

    An R-square value of 0.515 indicates that the model explains 51.5% of the variance in the dependent variable.
    The adjusted R-squared value of 0.387 is slightly lower, suggesting that some independent 
variables may not contribute much to the explanatory power of the model.
    The F-statistic of 4.023 and the associated p-value of 1.31e-127 suggest that the overall model is
statistically significant.
    The model includes 2900 observations and the log likelihood is -24029.
    AIC and BIC are measures of model fit that take into account the number of parameters in the model.
The lower the AIC and BIC values, the better the model fit. The AIC value is 4.927e+04 and the
BIC is equal to 5.289e+04.
    Finally, the "nonrobust" covariance type indicates that the standard errors of the coefficients are assumed
homoscedastic (i.e. constant at all levels of independent variables) and normal distributed.
    The omnibus test is a test of the normality of the residuals of a regression model. Value 1095.226
is the test statistic and the associated p-value is less than 0.05, as indicated by “Probability
(omnibus): 0.000." This suggests that the residuals are not normally distributed, which may be the reason
for concerns about model validity.
    The Durbin-Watson test is a test of autocorrelation of residuals. A value of 1.810 indicates that
there is some positive autocorrelation, although it is not strong enough to make model is invalid.
    The Jacques-Beer (JB) test is another test for the normality of residuals, and its statistical significance
the test value of 9370.045 and the associated p value of 0.00 indicate that the residuals are not distributed
Fine.
    A skewness value of 1.552 suggests that the distribution of residuals is positively skewed,
and the kurtosis value of 11.241 indicates that the distribution is heavy-tailed.
    The last line shows the condition number of the matrix of independent variables. High number
conditionality indicates that the model may have problems with multicollinearity, which may
make the coefficients difficult to interpret. The value of 1.33e+03 is relatively low, which indicates that
that multicollinearity may not be a significant problem in this model. '''
