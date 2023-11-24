import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from machine_learning_4 import y_test, y_knn_pred, y_regression_pred, y_best_tree_regression_pred, \
    y_best_elasticnet_pred, y_gradientboost_pred, y_adaboost_pred, y_blending_pred, y_polynomial_pred, y_rf_pred, \
    y_xgb_pred, y_bagging_pred, rmse_mean_xgb_regression, rmse_scores_xgb_regression

# 1) KNeighborsRegressor
# 2) LinearRegression
#3) DecisionTreeRegressor with tuned hyperparameters
#4) ElasticNet with tuned hyperparameters
# 5) LinearRegression with PolynomialFeatures
# 7) GradientBoostingRegressor
# 8) AdaBoostRegressor
# 11) BlendingRegressor

models = [
    ('KNN Regressor', y_knn_pred, 'red'),
    ('Linear Regression', y_regression_pred, 'green'),
    ('DecisionTreeRegressor', y_best_tree_regression_pred, 'blue'),
    ('ElasticNetRegressor', y_best_elasticnet_pred, 'orange'),
    ('PolynomialFeatures', y_polynomial_pred, 'purple'),
    ('GradientBoostingRegressor', y_gradientboost_pred, 'brown'),
    ('AdaBoostRegressor', y_adaboost_pred, 'pink'),
    ('BlendingRegressor', y_blending_pred, 'gray')
]

# А) Scatter graphs of predicted and actual prices
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
fig.subplots_adjust(hspace=1.2, wspace=0.8)
axs = axs.ravel()

for i, (model_name, y_pred, color) in enumerate(models):
    sns.scatterplot(x=y_test, y=y_pred, ax=axs[i], c=color)
    axs[i].plot([0, 10000], [0, 10000], 'k--')
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_title(model_name)
fig.suptitle('Regression models by actual (x-axis) and predicted (y-axis) values')
plt.tight_layout()
plt.show()

# С) Error distribution graphs
fig, ax = plt.subplots(figsize=(18, 8))
for i, (model_name, y_pred, color) in enumerate(models):
    distribution = y_pred - y_test
    sns.kdeplot(distribution, label=model_name, color=color)

ax.set_xlabel('Error')
ax.set_ylabel('Frequency')
ax.set_title('Regression models by error distribution')
ax.legend()

plt.show()

# Three best models:
# 10) RandomForestRegressor
# 13) xgb.XGBRegressor
# 14) BaggingRegressor

best_models = [
    ('RandomForestRegressor', y_rf_pred, 'red'),
    ('XGBoost Regressor', y_xgb_pred, 'green'),
    ('BaggingRegressor', y_bagging_pred, 'blue')
]

results = []
for best_model_name, y_pred, color in best_models:
    distribution = y_pred - y_test
    for error in distribution:
        results.append((best_model_name, y_test, error))

results_df = pd.DataFrame(results, columns=['model', 'y_test', 'distribution'])

# С) Error distribution graphs
fig, ax = plt.subplots(figsize=(18, 8))
for i, (best_model_name, y_pred, color) in enumerate(best_models):
    distribution = y_pred - y_test
    sns.kdeplot(distribution, label=best_model_name, color=color)

ax.set_xlabel('Error')
ax.set_ylabel('Frequency')
ax.set_title('Regression models by error distribution')
ax.legend()

plt.show()

# D) regplot error distributions
fig, axs = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(hspace=1.2, wspace=1.5)
axs = axs.ravel()

for i, (best_model_name, y_pred, color) in enumerate(best_models):
    sns.regplot(x=y_test, y=y_pred-y_test, ax=axs[i], scatter=True, color=color)
    axs[i].set_title(best_model_name)
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_ylim([-5000, 5000])

fig.suptitle('Regression models by frequency (ax y) of errors(ax x)')
plt.tight_layout()
plt.show()

# Visualization of cross-validation of the best model
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(range(len(rmse_scores_xgb_regression)), rmse_scores_xgb_regression, color='blue')
ax.set_yticks(range(len(rmse_scores_xgb_regression)))
ax.set_yticklabels(['Fold {}'.format(i+1) for i in range(len(rmse_scores_xgb_regression))])
plt.axvline(rmse_mean_xgb_regression, color='y', linestyle='dashed', linewidth=2.5,
            label=f"RMSE: {round(rmse_mean_xgb_regression)}")
for i, v in enumerate(rmse_scores_xgb_regression):
    plt.text(v+7, i, f"{round(v)}", ha='left', va='center', color="black", fontsize=12,
             bbox=dict(facecolor='orange', edgecolor='pink', boxstyle='round4,pad=0.4'))
ax.set_xlim(0, rmse_mean_xgb_regression.max() * 1.225)
plt.xlabel('RMSE')
plt.ylabel('Fold')
plt.title('Cross-validation results of XGBoost Regression')
plt.legend(loc='upper right', fontsize=13)
plt.gca().grid(False)

plt.show()
