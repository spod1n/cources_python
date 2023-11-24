''' 3. Based on the EDA and your common sense, choose two hypotheses you
want to test/analyse. For each hypotheses list the null hypothesis and other possible
alternative hypotheses, design tests to distinguish between them, and complete them.
Describe the results. '''

''' Hypothesis 1. The price of furniture created by popular designers is higher than that of furniture created by
designed by lesser known designers.
    We can define popular designers as those whose names appear in more than 50% of the set
data, and lesser-known designers as those whose names appear in less than 50% of the data set.
    Null hypothesis: there is no significant difference in the price of furniture between popular and less well-known
designers do not.
    Alternative hypothesis 1: furniture designed by popular designers is more expensive than furniture
designed by lesser-known designers.
    Alternative Hypothesis 2: Furniture designed by lesser-known designers is more expensive than furniture designed by lesser-known designers.
designed by popular designers. '''

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import ranksums
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from analysis_data_1 import analysis_data

digit_mask = analysis_data['designer'].str.contains('\d')
designers_data = analysis_data[~digit_mask].reset_index(drop=True)

designers_counts = designers_data['designer'].value_counts()[::-1]

# Calculate the sum of counts
sum_of_counts = designers_counts.sum()

# Calculate the cumulative sum of percentages
cumulative_percentage = designers_counts.cumsum() / sum_of_counts

# Find the designers that contribute to 50% of the total counts
famous_designers = designers_counts[cumulative_percentage > 0.5][::-1]

print(f'Famous Designers contributing to 50% of total counts:')
print(famous_designers)

famous_designers_data = designers_data[designers_data['designer'].isin(famous_designers.index)]
less_known_designers_data = designers_data[~designers_data['designer'].isin(famous_designers.index)]

#1) t-test to compare the average prices of two groups
t_statistic, p_value = ttest_ind(famous_designers_data['price'], less_known_designers_data['price'],
                                 equal_var=False)
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

famous_designers_mean = np.mean(famous_designers_data['price'])
less_known_designers_mean = np.mean(less_known_designers_data['price'])
difference_means = famous_designers_mean - less_known_designers_mean
if difference_means > 0:
    print(''' Alternative hypothesis 1 is confirmed: furniture designed by popular designers, "
        more expensive than furniture designed by lesser-known designers. ''')
else:
    print(''' Alternative hypothesis 2 is confirmed: furniture designed by lesser-known designers is more expensive
        furniture designed by popular designers. ''')

''' Alternative Hypothesis 2 is true for this hypothesis.
    A p value of 0.000 indicates that the probability of observing such a large difference by chance is
in the mean values between the two groups is very small. In other words, the null hypothesis stating
that there is no difference between the average prices of the products developed by the two groups can be rejected
with a high degree of reliability. '''

# 2) Mann-Whitney-Wilcoxon test
statistic, p_value = mannwhitneyu(famous_designers_data['price'], less_known_designers_data['price'])

print(f"Statistic: {statistic:.2f}")
print(f"P-value: {p_value:.3f}")

''' If the Mann-Whitney U test yielded a p-value of 0.000, this means that the test detected
a significant difference in the average price of furniture between well-known and lesser-known designers.
Since the p-value is less than the typical significance level of 0.05, we can reject the null hypothesis
and conclude that there is a statistically significant difference in the median price of furniture between 
the two groups.
    Thus, the Mann-Whitney U test and the difference in average price values (see the first test) support
alternative hypothesis 2. '''

# 3) Bootstrap Test
n_bootstraps = 10000
diffs = np.empty(n_bootstraps)
for i in range(n_bootstraps):
    famous_samples = np.random.choice(famous_designers_data['price'], size=len(famous_designers_data), replace=True)
    less_known_samples = np.random.choice(less_known_designers_data['price'], size=len(less_known_designers_data),
                                          replace=True)
    diffs[i] = np.mean(famous_samples) - np.mean(less_known_samples)

p_value = np.sum(diffs >= 0) / n_bootstraps
print(f"P-value: {p_value:.3f}")

''' So, Bootstrap Test and the difference in average price values (see the first test) support
alternative hypothesis 2. '''

#4) Shuffle test
famous_prices_values = famous_designers_data['price'].values
less_known_prices_values = less_known_designers_data['price'].values

observed_diff = np.mean(famous_prices_values) - np.mean(less_known_prices_values)

concatenated_prices = np.concatenate([famous_prices_values, less_known_prices_values])
np.random.shuffle(concatenated_prices)

shuffled_famous_prices = concatenated_prices[:len(famous_prices_values)]
shuffled_less_known_prices = concatenated_prices[len(famous_prices_values):]

shuffled_diff = np.mean(shuffled_famous_prices) - np.mean(shuffled_less_known_prices)

n_permutations = 1000
null_distribution = np.zeros(n_permutations)
for i in range(n_permutations):
    np.random.shuffle(concatenated_prices)
    shuffled_famous_prices = concatenated_prices[:len(famous_prices_values)]
    shuffled_less_known_prices = concatenated_prices[len(famous_prices_values):]
    null_distribution[i] = np.mean(shuffled_famous_prices) - np.mean(shuffled_less_known_prices)

p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))

print("Observed difference in mean values: {:.2f}".format(observed_diff))
print("P-value: {:.4f}".format(p_value))

''' Based on the observed mean difference of -243.64 and p-value of 0.0000, we can
reject the null hypothesis and conclude that there is a significant difference in the 
price of furniture between popular and lesser-known designers. Moreover, alternative 
hypothesis 2 is supported, according to which furniture created by lesser-known designers 
is more expensive than furniture, designed by popular designers. '''

# 5) Tukey's honestly significant difference (HSD) test
''' The Tukey HSD test is a commonly used multiple comparison test that compares
all possible pairs of group means and calculates a confidence interval for the 
difference in means. If the confidence interval does not include zero, it suggests 
a significant difference between the two in groups. '''

famous_prices = famous_designers_data['price']
less_known_prices = less_known_designers_data['price']

tukey_results = pairwise_tukeyhsd(np.concatenate([famous_prices, less_known_prices]),
                                  np.concatenate([['famous'] * len(famous_prices),
                                                  ['less-known'] * len(less_known_prices)]))

print(tukey_results.summary())

''' The result shows that the average difference between the "famous" group and the "less famous" group
is 243.6383 with a p-value of 0.0. This means that the average price of products developed by well-known
designers, significantly lower than the average price of products designed by lesser-known designers.
    The lower and upper limits of the confidence interval are 140.5739 and 346.7026, respectively.
Since the confidence interval does not contain zero, we can reject the null hypothesis and conclude,
that the difference in average prices between the two groups is statistically significant at the 0.05 
significance level. '''

# 6) ANOVA test - analysis of variance
''' This is a parametric test that can be used to compare averages over
than two groups. In our case, we can use it to compare average prices
for furniture designed by popular designers and lesser-known designers. '''

_, p_value = stats.f_oneway(famous_prices, less_known_prices)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "between popular and lesser-known designers.")
else:
    print("The null hypothesis could not be rejected. Significant difference in the average price of furniture "
          "there is no difference between popular and lesser-known designers")

''' So, the ANOVA test and the difference in average prices (see the first test) support
alternative hypothesis 2. '''

# 7 Wilcoxon signed-rank test
min_length = min(len(famous_prices_values), len(less_known_prices_values))
famous_prices_values = famous_prices_values[:min_length]
less_known_prices_values = less_known_prices_values[:min_length]

statistic, p_value = wilcoxon(famous_prices_values, less_known_prices_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

''' The Wilcoxon signed rank test was conducted for two samples of furniture prices (furniture from famous
designers and furniture from lesser-known designers) to test whether there is a significant difference
in average prices between the two groups. The test resulted in a test statistic of 353068.00 and a p-value of 0.0000.
    The test statistic is the sum of the ranks assigned to the differences between each pair
observations in two samples. The larger the test statistic, the stronger the evidence against the null.
hypothesis which states that there is no difference between the means of two groups.
    The p-value is the probability of observing a test statistic as extreme as
obtained under the null hypothesis. In this case, the p-value is very small (less than 0.001), which means
that the probability of observing such a large test statistic by chance is very small. Thus,
we reject the null hypothesis and conclude that there is a significant difference in average prices between the two
groups of furniture.
    So, the Wilcoxon signed rank test and the difference in mean prices (see first test) support
alternative hypothesis 2. '''

# 8) Wilcoxon rank-sum test
statistic, p_value = ranksums(famous_prices_values, less_known_prices_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

''' Unlike the previous test, the test statistics are based on the ranks of values in two samples, and
not on the differences between values.
    The results of the Wilcoxon rank sum test show that there is a significant difference
between the prices of furniture from popular and lesser-known designers. The test statistics is -8.48,
and the p-value is 0.0000, which indicates that the probability of such a random observation
There is very little difference in the averages. Therefore, we can reject the null hypothesis and
conclude that prices for furniture designed by popular designers differ significantly
from the prices of furniture designed by lesser-known designers.
    So, the Wilcoxon rank sum test and the difference in mean prices (see first test) support
alternative hypothesis 2. '''

# 9) Kruskal-Wallis test
statistic, p_value = kruskal(famous_designers_data['price'], less_known_designers_data['price'])

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

''' The test statistic was calculated as 40.56 and the p-value was found to be 0.0000.
This indicates that there is good reason to reject the null hypothesis.
    So, the Kruskal-Wallis test and the difference in average price values (see the first test) support
alternative hypothesis 2. '''

# 10) Robust logistic regression - Устойчивая логистическая регрессия
designers_data['famous_designer'] = (designers_data['designer'].isin(famous_designers)).astype(int)

X = designers_data['famous_designer']
y = designers_data['price']

X = sm.add_constant(X)

model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
results = model.fit()
print(results.summary())

if results.pvalues[0] > 0.05:
    print("There is no fundamental difference in average prices.")
else:
    print("There is a significant difference in average prices.")

''' So, robust logistic regression and mean price difference (see first test) support
alternative hypothesis 2. '''

# 11) McNemar's test
table = np.array([famous_prices_values, less_known_prices_values])

result = mcnemar(table, exact=True)

print(f"McNemar's test statistic: {result.statistic:.2f}")
print(f"P-value: {result.pvalue:.3f}")

if result.pvalue < 0.05:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "between popular and lesser-known designers.")
else:
    print("The null hypothesis could not be rejected. Significant difference in the average price of furniture "
          "there is no difference between popular and lesser-known designers")

''' So, McNemar's test and the difference in average prices (see the first test) support
alternative hypothesis 2. '''

''' Overall conclusion: tests support alternative hypothesis 2: furniture designed by lesser known
by designers, more expensive than furniture designed by popular designers. '''
