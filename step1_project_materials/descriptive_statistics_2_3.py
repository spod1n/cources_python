# 3. Prices and product categories

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from analysis_data_1 import analysis_data

# Product information: product categories and number of items by category
print(analysis_data["category"].unique())
print(analysis_data["category"].value_counts())

# Or
value_counts_by_category = analysis_data.groupby("category")['price'].agg(["count"]).reset_index()
value_counts_by_category = value_counts_by_category.sort_values('count', ascending=False).reset_index(drop=True)
print(value_counts_by_category)

plt.figure(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
plt.bar(value_counts_by_category['category'], value_counts_by_category['count'],
        color=gradient_colors(value_counts_by_category['count'] / value_counts_by_category['count'].max()))

plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(value_counts_by_category['count']):
    plt.text(i, v + 15, str(v), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.01'))

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Number of titles by Categories")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y', linestyle='--')
plt.ylim(0, value_counts_by_category['count'].max() * 1.1)
plt.subplots_adjust(bottom=0.32, left=0.1)

plt.show()

# Descriptive statistics for price by category
print(analysis_data.groupby("category")['price'].describe())

# Or (let's highlight the most significant values for us)
price_stats = analysis_data.groupby("category")["price"].agg(["max", "mean", "median", "min"]).reset_index()
price_stats = price_stats.sort_values('max', ascending=False).reset_index(drop=True)
print(price_stats)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(price_stats['category'], price_stats["max"], label="Max")
ax.bar(price_stats['category'], price_stats["mean"], label="Mean", alpha=0.7)
ax.bar(price_stats['category'], price_stats["median"], label="Median")
ax.bar(price_stats['category'], price_stats["min"], label="Min", alpha=0.7)
plt.ylim(0, price_stats["max"].max() * 1.1)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Prices by Categories")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.grid(axis='y', linestyle='--')
for i, v in enumerate(price_stats["max"]):
    plt.text(i, v+300, str(round(v)), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.1'))
plt.subplots_adjust(bottom=0.32, left=0.1)

plt.show()

# Median price depending on category
median_price = analysis_data.groupby("category")["price"].median().reset_index()
median_sort_price = median_price.sort_values('price', ascending=False).reset_index(drop=True)
print(median_sort_price)

fig, ax = plt.subplots(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

ax.bar(median_sort_price['category'], median_sort_price['price'],
       color=gradient_colors(median_sort_price['price']/median_sort_price['price'].max()),
       width=0.5, edgecolor='black', linewidth=1, alpha=0.7, align='center',
       capsize=7, ecolor='gray', hatch='///',
       error_kw=dict(elinewidth=1, capsize=3, capthick=1))

ax.set_xticks(range(len(median_sort_price['category'])))
ax.set_xticklabels(median_sort_price['category'], rotation=45, ha='right', fontsize=10)
ax.set_title("Median prices by Categories")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(150, median_sort_price['price'].max() * 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i, v in enumerate(median_sort_price['price']):
    ax.text(i, v+60, str(round(v)), ha='center', va='center', color="b",
            bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.1'))

plt.subplots_adjust(bottom=0.32, left=0.1)

plt.show()

formula = "price ~ category"
model = smf.ols(formula=formula, data=analysis_data).fit()
print(model.summary())

''' Result of multiple linear regression model using conventional method
Ordinary Least Squares (OLS).
    The model includes 16 predictor variables and one dependent variable (price).

    The R-squared value of the model is 0.159, which means that approximately 15.9% of the variance in the dependent
variable can be explained by the predictor variables in the model.
    The adjusted R-squared value is 0.154, which takes into account the number
predictor variables in the model and adjusts the R-squared value accordingly.
    The F-statistic is 34.05 with a corresponding probability value (Prob) of 3.08e-96. This indicates
there is strong evidence that at least one of the predictor variables is significantly
associated with the dependent variable.
    The log likelihood value of the model is -24827, which is used for comparison
models and assessment of their compliance. Akaike Information Criterion (AIC) and Bayesian Information Criterion
criterion (BIC) also help to compare the current model with others.
    The covariance type is unreliable, which means the model is sensitive to outliers and influential observations.
Overall, the model has a low R-squared value, indicating that the predictor variables are not
are strong predictors of the dependent variable. The F-statistic is significant, indicating
that there is a significant relationship between the predictor variables and the dependent variable,
but the effect size is relatively small.
    The omnibus test is a test of the normality assumption of residuals in a linear regression model.
A value of 1299.027 indicates that the residuals are not normally distributed, with a corresponding value
probability (Prob(Omnibus)) equal to 0.000. This violates one of the assumptions of the linear regression model and
may affect the reliability of the results.
    The Durbin-Watson statistic is a test for the presence of autocorrelation in the residuals. Value 1.708
indicates that there may be some positive autocorrelation in the residuals, but not that much
serious enough to invalidate the results.
    The Jarque-Bera (JB) test is another test of normality, but it takes into account both skewness and kurtosis.
A value of 7022.093 and a probability (Prob(JB)) of 0.00 indicate that the residuals are not normally distributed,
which confirms the results of the omnibus test.
    A skewness value of 2.101 indicates that the distribution of residuals is positively skewed,
and a kurtosis value of 9.360 indicates that the distribution is more peaked than normal
distribution.
    The condition number (Cond. No.) of 35.5 is a measure of multicollinearity in the model.
A higher value indicates a greater degree of multicollinearity, which may make it difficult to
interpretation of individual coefficients. However, a value of 35.5 suggests that multicollinearity
is not a major problem in this model. '''
