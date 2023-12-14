''' Hypothesis 2. Seating Furniture is on average more expensive than storage furniture
(Storage Furniture), i.e. there is a significant difference in average price depending on function
(use) furniture.
    Seating furniture: Sofas & armchairs, Beds, Chairs, Children's furniture, Nursery furniture,
Café furniture, Bar furniture, Outdoor furniture.
    Storage furniture: Tables & desks, Trolleys, Wardrobes, Bookcases & shelving units,
TV & media furniture, Cabinets & cupboards, Room dividers, Sideboards, buffets & console tables,
Chests of drawers & drawer units.
    Null hypothesis: The average price of seating and storage furniture is the same.
    Alternative hypothesis 1: The average price of seating furniture is higher than storage furniture.
    Alternative Hypothesis 2: The average price of storage furniture is higher than that of seating furniture. '''

import np as np
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, ranksums, kruskal
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from analysis_data_1 import analysis_data

category_groups = {
    'Seating Furniture': ['Sofas & armchairs', 'Beds', 'Chairs', 'Children\'s furniture', 'Nursery furniture',
                                   'Café furniture', 'Bar furniture', 'Outdoor furniture'],
    'Storage Furniture': ['Tables & desks','Trolleys', 'Wardrobes', 'Bookcases & shelving units',
                          'TV & media furniture', 'Cabinets & cupboards', 'Room dividers',
                          'Sideboards, buffets & console tables', 'Chests of drawers & drawer units']
}

grouped_data = analysis_data.groupby('category')

seating_furniture = grouped_data.get_group('Sofas & armchairs')
for category in category_groups['Seating Furniture'][1:]:
    seating_furniture = pd.concat([seating_furniture, grouped_data.get_group(category)])

storage_furniture = grouped_data.get_group('Tables & desks')
for category in category_groups['Storage Furniture'][1:]:
    storage_furniture = pd.concat([storage_furniture, grouped_data.get_group(category)])

print('Seating Furniture:')
print(seating_furniture)

print('\nStorage Furniture:')
print(storage_furniture)

#1) t-test to compare the average prices of two groups
t_statistic, p_value = ttest_ind(seating_furniture['price'], storage_furniture['price'],
                                 equal_var=False)
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

seating_furniture_mean = np.mean(seating_furniture['price'])
storage_furniture_mean = np.mean(storage_furniture['price'])
difference_means = seating_furniture_mean - storage_furniture_mean
if difference_means > 0:
    print(''' Alternative hypothesis 1 is confirmed: the average price of seating furniture is higher,
            than for storage. ''')
else:
    print(''' Alternative hypothesis 2 is confirmed: the average price of furniture for storage is higher,
            than for sitting. ''')


''' Since the p-value is very small (p-value <0.05), we can reject the null hypothesis,
which states that the average price of seating and storage furniture is the same. Hence,
we can conclude that there is a significant difference in the average price depending on 
the function (use) of the furniture. Therefore, Alternative Hypothesis 1 is true and we 
can say that the seating furniture on average, it costs more than storage furniture. '''

# 2) Mann-Whitney-Wilcoxon test
statistic, p_value = mannwhitneyu(seating_furniture['price'], storage_furniture['price'])

print(f"Statistic: {statistic:.2f}")
print(f"P-value: {p_value:.3f}")

''' According to the results of the Mann-Whitney U test, the p-value is less than 0.05, which indicates a significant difference
in prices for seating and storage furniture. Alternative hypothesis 1 that the average price of furniture
for seating is higher than for storage furniture, confirmed by the Mann-Whitney U-test and the difference in means
price values (see first test). Therefore, we can reject the null hypothesis that
The average price of seating and storage furniture is the same. '''

# 3) Bootstrap Test
def diff_means(x, y):
    return np.mean(x) - np.mean(y)

n_iterations = 1000
differences = []
for i in range(n_iterations):
    sample_a = resample(seating_furniture['price'], replace=True, n_samples=len(seating_furniture))
    sample_b = resample(storage_furniture['price'], replace=True, n_samples=len(storage_furniture))
    difference = diff_means(sample_a, sample_b)
    differences.append(difference)

alpha = 0.05
lower = np.percentile(differences, alpha/2*100)
upper = np.percentile(differences, (1-alpha/2)*100)

if lower <= 0 <= upper:
    print("There is no fundamental difference in average prices.")
else:
    print("There is a significant difference in average prices.")

''' So, Bootstrap Test and the difference in average price values (see the first test) support
alternative hypothesis 1. '''

# 4) Shuffle test
observed_diff = np.mean(seating_furniture['price']) - np.mean(storage_furniture['price'])

n_iterations = 1000
differences = []
combined_data = pd.concat([seating_furniture, storage_furniture])

for i in range(n_iterations):
    np.random.shuffle(combined_data['price'].values)
    sample_a = combined_data.iloc[:len(seating_furniture)]
    sample_b = combined_data.iloc[len(seating_furniture):]
    difference = np.mean(sample_a['price']) - np.mean(sample_b['price'])
    differences.append(difference)

p_value = (np.abs(differences) >= np.abs(observed_diff)).mean()

print("Observed difference in means: {:.2f}".format(observed_diff))
print("P-value: {:.4f}".format(p_value))

if p_value < 0.05:
    print("There is a significant difference in average prices between seating furniture "
          "and furniture for storage, (p = {:.3f}).".format(p_value))
else:
    print("There is no significant difference in average prices between seating and furniture "
          "for storage, (p = {:.3f}).".format(p_value))

''' The results of the permutation test show that there is a significant difference between the means
values of the groups “Furniture for seating” and “Furniture for storage”. In particular, the average price of furniture
for seating is $413.14 higher than for storage furniture, with a p value less than 0.05, which
provides strong evidence against the null hypothesis of no difference. '''

# 5) Tukey's honestly significant difference (HSD) test
results = pairwise_tukeyhsd(
np.concatenate([seating_furniture['price'], storage_furniture['price']]),
np.concatenate([['Seating Furniture']*len(seating_furniture), ['Storage Furniture']*len(storage_furniture)]),
alpha=0.05)

print(results)

''' The Tukey HSD test results show that there is a significant difference between the means
values of the groups “Furniture for seating” and “Furniture for storage”. In particular, the average price of furniture
for seating is $413.14 higher than for storage furniture, with confidence interval
from $512.55 to $313.74 and a p value less than 0.05, providing strong evidence
against the null hypothesis of no difference. '''

# 6) ANOVA test - analysis of variance
_, p_value = stats.f_oneway(seating_furniture['price'], storage_furniture['price'])

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")
else:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same")

''' So, the ANOVA test and the difference in average prices (see the first test) support
alternative hypothesis 1. '''

# 7) Wilcoxon signed-rank test
seating_values = seating_furniture['price'].values
storage_values = storage_furniture['price'].values
min_length = min(len(seating_values), len(storage_values))
seating_values = seating_values[:min_length]
storage_values = storage_values[:min_length]

statistic, p_value = wilcoxon(seating_values, storage_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

if p_value > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, the Wilcoxon signed-rank test and the difference in average price values (see the first test) support
alternative hypothesis 1. '''

# 8) Wilcoxon rank-sum test
statistic, p_value = ranksums(seating_values, storage_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

if p_value > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, the Wilcoxon rank-sum test and the difference in average price values (see the first test) support
alternative hypothesis 1. '''

# 9) Kruskal-Wallis test
statistic, p_value = kruskal(seating_furniture['price'], storage_furniture['price'])

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

if p_value > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, the Kruskal-Wallis test and the difference in average price values (see the first test) support
alternative hypothesis 1. '''

# 10) Robust logistic regression

X = pd.concat([seating_furniture['price'], storage_furniture['price']])
y = np.concatenate([np.ones(len(seating_furniture)), np.zeros(len(storage_furniture))])

model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
results = model.fit()
print(results.summary())

if results.pvalues[0] > 0.05:
    print("There is no fundamental difference in average prices.")
else:
    print("There is a significant difference in average prices.")

''' So, robust logistic regression and mean price difference (see first test) support
alternative hypothesis 1. '''

# 11) McNemar's test
table = np.array([seating_values, storage_values])

result = mcnemar(table, exact=True)

print(f"McNemar's test statistic: {result.statistic:.2f}")
print(f"P-value: {result.pvalue:.3f}")

if result.pvalue > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, McNemar's test and the difference in average prices (see the first test) support
alternative hypothesis 1. '''

# 12) Poisson regression
X = seating_values
y = storage_values

model = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())
results = model.fit()

print(results.summary())

if results.pvalues[1] > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

"""    So, the Poisson regression model and the difference in mean prices (see first test) support
alternative hypothesis 1. """

# 13) Negative binomial regression
X = seating_values
y = storage_values

model = sm.GLM(y, sm.add_constant(X), family=sm.families.NegativeBinomial(alpha=0.5))
results = model.fit()

print(results.summary())

if results.pvalues[1] > 0.05:
    print("The null hypothesis could not be rejected. Average price of seating and storage furniture "
          "is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

'''   So, negative binomial regression and the difference in average prices (see first test)
support alternative hypothesis 1. '''

# 14) Zero-inflated regression model
X = seating_values
y = storage_values

zip_model = sm.ZeroInflatedNegativeBinomialP(y, sm.add_constant(X)).fit()

print(zip_model.summary())

if zip_model.pvalues[1] > 0.05:
    print("The null hypothesis could not be rejected. Average price of seating and storage furniture "
          "is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

'''   So, the regression model with zero inflation and the difference in average prices (see the first test)
support alternative hypothesis 1. '''

# 15) OLS regression model
X = pd.concat([seating_furniture['price'], storage_furniture['price']])
y = np.concatenate([np.ones(len(seating_furniture)), np.zeros(len(storage_furniture))])

model = sm.OLS(y, sm.add_constant(X))
results = model.fit()
print(results.summary())

if results.pvalues[1] > 0.05:
    print("There is no fundamental difference in average prices.")
else:
    print("There is a significant difference in average prices.")

''' The OLS regression model shows that there is a significant difference in the average price between
seating furniture and storage furniture with a p-value of 5.38e-16. Coefficient of the variable “price”
positive, indicating that seating furniture is generally more expensive than outdoor furniture
storage However, the R-squared value is low at 0.022, indicating that the model
does not explain most of the variance in the data. The model intercept test is also significant at p-value,
equal to 0.000, indicating that there is a non-zero price difference between seating furniture
and furniture for storage. '''

''' Overall conclusion: tests confirm alternative hypothesis 1: the average price of seating furniture is higher,
than for storage. '''
