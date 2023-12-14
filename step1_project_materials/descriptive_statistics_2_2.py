# 2. Prices and designers

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

from analysis_data_1 import analysis_data

# Let's prepare the data set for descriptive statistics on price depending on the designer's name.
# Weed out unrecognizable designer names
digit_mask = analysis_data['designer'].str.contains('\d')
digit_designer_data = analysis_data[~digit_mask].reset_index(drop=True)
print(digit_designer_data)

# Product information: designer names and number of items by designer name
print(digit_designer_data["designer"].unique())
value_counts_by_designer = digit_designer_data.groupby("designer")["price"].agg(["count"]).reset_index()
value_counts_by_designer = value_counts_by_designer[value_counts_by_designer['count'] >= 30]\
    .sort_values('count', ascending=False).reset_index(drop=True)
print(value_counts_by_designer)

plt.figure(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
plt.bar(value_counts_by_designer['designer'], value_counts_by_designer['count'],
        color=gradient_colors(value_counts_by_designer['count'] / value_counts_by_designer['count'].max()))

plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(value_counts_by_designer['count']):
    plt.text(i, v + 20, str(v), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.01'))

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Number of titles by Designers")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y', linestyle='--')
plt.ylim(10, value_counts_by_designer['count'].max() * 1.1)
plt.subplots_adjust(bottom=0.3, left=0.1)

plt.show()

# Descriptive statistics for price by designer name
print(digit_designer_data.groupby("designer")['price'].describe())

# Median price depending on designer name
median_price_designer = digit_designer_data.groupby("designer")["price"].median().reset_index()
median_price_designer = median_price_designer.sort_values('price', ascending=False).reset_index(drop=True)
median_price_designer = median_price_designer.iloc[:16]
print(median_price_designer)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FF7F00", "#FFFF00", "#00FF00"])
bar_heights = np.arange(len(median_price_designer))
bar_widths = median_price_designer['price']
colors = cmap(median_price_designer['price']/median_price_designer['price'].max())

ax.barh(bar_heights, bar_widths, color=colors)

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax)

ax.set_xlim(150, median_price_designer['price'].max() * 1.12)
ax.set_yticks(bar_heights)
ax.set_yticklabels(median_price_designer['designer'])
plt.title("Median price by Designers")
plt.ylabel("")
plt.xlabel("")

plt.grid(axis='x', linestyle='--', alpha=0.5)

gradient_cmap = mcolors.LinearSegmentedColormap.from_list("", ["#002FFF", "#FF0000"])
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
im = ax.imshow(gradient, cmap=gradient_cmap, aspect='auto', extent=[150, median_price_designer['price'].max() * 1.12,
                                                                    -0.5, len(median_price_designer)-0.5])

for i, v in enumerate(median_price_designer['price']):
    ax.text(v + 450, i, str(round(v)), ha='center', va='center', color="black", fontsize=11,
            bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.05'))

plt.subplots_adjust(left=0.4, bottom=0.1)
plt.gca().invert_yaxis()
plt.show()

formula = "price ~ designer"
model = smf.ols(formula=formula, data=digit_designer_data).fit()
print(model.summary())

''' Result of multiple linear regression model using conventional method Ordinary Least Squares (OLS).
    The model includes 196 predictor variables and one dependent variable (price).

    The R-squared value of the model is 0.448, which means that approximately 44.8% of the variance in the dependent
variable can be explained by the predictor variables in the model.
    The adjusted R-squared value is 0.407, which takes into account the number of predictor variables
in the model and adjusts the R-squared value accordingly.
    The F-statistic is 10.79 with a corresponding probability value (Prob) of 7.46e-219.
    This indicates that there is strong evidence that at least one of the predictor variables
significantly associated with the dependent variable.
    The log likelihood value of the model is -23395, which is used to compare models
and assessment of their compliance.
    Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) also help
compare the current model with others.
    It is important to note that the covariance type is unstable, meaning that the model is sensitive to outliers.
    Omnibus Test - A value of 1331.561 indicates that the residuals are not normally distributed,
with a corresponding probability value (Prob(Omnibus)) of 0.000. This violates one of the assumptions
linear regression models and may affect the reliability of the results.
    The Durbin-Watson statistic is a test for the presence of autocorrelation in the residuals. Value 1.817
indicates that there may be some positive autocorrelation in the residuals, but not that much
serious enough to invalidate the results.
    The Jarque-Bera (JB) test is another test of normality, but it takes into account both skewness and kurtosis.
A value of 10710.006 and a probability (Prob(JB)) of 0.00 indicate that the residuals are not normally distributed,
which confirms the results of the Omnibus test.
    A skewness value of 2.092 indicates that the distribution of residuals has a positive
skewness, and the kurtosis value of 11.623 indicates that the distribution has heavy tails
and is more peaked than the normal distribution.
    Condition number (Cond. No.) 146 is a measure of multicollinearity in the model. Higher value
indicates a high degree of multicollinearity, which may make it difficult to interpret individual
coefficients However, a value of 146 suggests that multicollinearity is not severe
problem in this model. '''
