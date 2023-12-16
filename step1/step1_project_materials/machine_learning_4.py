''' Train a model to predict the price of furniture.
- Indicate which columns should not be included in the model and why.
- Create a cross-validation pipeline for training and evaluation
models, including (if necessary) steps such as imputation
missing values and normalization.
- Suggest methods to improve model performance.
Describe the results. '''

from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from analysis_data_1 import analysis_data

''' Before training the model, we must exclude unnecessary columns such as 'index', 'item_id',
since they do not affect the price forecast.
    Additionally, we should exclude the "old_price" column since it is highly correlated with
"price" column, and using both functions may cause multicollinearity problems. '''

exclude_cols = ['item_id', 'old_price', 'price']
if 'index' in analysis_data.columns:
    analysis_data = analysis_data.drop('index', axis=1)

num_cols = ['depth', 'height', 'width']
cat_cols = ['name', 'category', 'designer', 'sellable_online',  'other_colors',
            'link', 'short_description']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

X = analysis_data.drop(exclude_cols, axis=1)
y = analysis_data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 1) KNeighborsRegressor
model_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn_regressor', KNeighborsRegressor())
])

model_knn.fit(X_train, y_train)

y_knn_pred = model_knn.predict(X_test)

''' Let's evaluate the performance of the model:
    RMSE is the square root of MSE (mean square error) and is more interpretable
value because it is in the same units as the forecast and actual values. '''

rmse_knn = np.sqrt(np.mean((y_knn_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_knn))

mse_knn = np.mean((y_knn_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_knn))

# 2) LinearRegression
model_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_regression.fit(X_train, y_train)

y_regression_pred = model_regression.predict(X_test)

rmse_regression = np.sqrt(np.mean((y_regression_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_regression))

mse_regression = np.mean((y_regression_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_regression))

""" The linear regression model performs slightly better than the K-nearest neighbors algorithm with
in terms of RMSE and root mean square error, indicating that on average it gives
more accurate forecasts. """

# 3) DecisionTreeRegressor
model_tree_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('tree_regressor', DecisionTreeRegressor(random_state=42))
])

model_tree_regression.fit(X_train, y_train)

y_tree_regression_pred = model_tree_regression.predict(X_test)

rmse_tree_regression = np.sqrt(np.mean((y_tree_regression_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_tree_regression))

mse_tree_regression = np.mean((y_tree_regression_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_tree_regression))

# Let's configure the DecisionTreeRegressor hyperparameters:
param_grid = {
    'tree_regressor__max_depth': [16, 20, 24, 28],
    'tree_regressor__min_samples_split': [2, 4, 6, 8],
    'tree_regressor__min_samples_leaf': [6, 8, 11, 15]
}

grid_search = GridSearchCV(
    estimator=model_tree_regression,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print('Best hyperparameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# DecisionTreeRegressor with customized hyperparameters
best_params = {
    'max_depth': 20,
    'min_samples_leaf': 8,
    'min_samples_split': 2,
    'splitter': 'best'
}

model_best_tree_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('best_tree_regressor', DecisionTreeRegressor(random_state=42, **best_params))
])

model_best_tree_regression.fit(X_train, y_train)

y_best_tree_regression_pred = model_best_tree_regression.predict(X_test)

rmse_best_tree_regression = np.sqrt(np.mean((y_best_tree_regression_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_best_tree_regression))

mse_best_tree_regression = np.mean((y_best_tree_regression_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_best_tree_regression))

''' The results show that the decision tree regressor has higher MSE and RMSE than
linear regression, which indicates that the decision tree model has a lower
performance in predicting the target variable (price) compared to linear regression.
This is probably due to the fact that decision trees can overfit the data and become
too complex, resulting in poor performance on new, unseen data.
On the other hand, linear regression is a simpler and more interpretable model that
can work well when there is a linear relationship between the input features and the 
target variable. '''

# 4) ElasticNet
model_elasticnet = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('elasticnet_regressor', ElasticNet())
])

model_elasticnet.fit(X_train, y_train)

y_elasticnet_pred = model_elasticnet.predict(X_test)

rmse_elasticnet = np.sqrt(np.mean((y_elasticnet_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_elasticnet))

mse_elasticnet = np.mean((y_elasticnet_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_elasticnet))

# Let's configure ElasticNet hyperparameters:
param_grid = {
    'elasticnet_regressor__alpha': [0.1, 0.5, 1, 5, 10],
    'elasticnet_regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}

grid_search = GridSearchCV(
    estimator=model_elasticnet,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print('Best hyperparameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# ElasticNet with tuned hyperparameters
best_params = {
    'elasticnet_regressor__alpha': 0.1,
    'elasticnet_regressor__l1_ratio': 0.9
}

model_best_elasticnet = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('best_elasticnet_regressor', ElasticNet())
])

model_best_elasticnet.fit(X_train, y_train)

y_best_elasticnet_pred = model_best_elasticnet.predict(X_test)

rmse_best_elasticnet = np.sqrt(np.mean((y_best_elasticnet_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_best_elasticnet))

mse_best_elasticnet = np.mean((y_best_elasticnet_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_best_elasticnet))

''' The ElasticNet model performed the worst because it has a more complex approach
to regularization compared to other models used in this analysis, which could lead to
to data retraining. '''

# 5) PolynomialFeatures
num_transformer_polynomial = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2))
])

preprocessor_polynomial = ColumnTransformer(transformers=[
    ('num_polynomial', num_transformer_polynomial, num_cols),
    ('cat', cat_transformer, cat_cols)
])

model_polynomial = Pipeline(steps=[
    ('preprocessor_polynomial', preprocessor_polynomial),
    ('regressor_polynomial', LinearRegression())
])

model_polynomial.fit(X_train, y_train)

y_polynomial_pred = model_polynomial.predict(X_test)

rmse_polynomial = np.sqrt(np.mean((y_polynomial_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_polynomial))

mse_polynomial = np.mean((y_polynomial_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_polynomial))

""" A linear regression model with polynomial functions performs slightly better than
other models in terms of RMSE and root mean square error, indicating that
on average it gives the most accurate forecasts. """

# 6) RandomForestRegressor
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf_regressor', RandomForestRegressor(random_state=42))
])

model_rf.fit(X_train, y_train)

y_rf_pred = model_rf.predict(X_test)

rmse_rf = np.sqrt(np.mean((y_rf_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_rf))

mse_rf = np.mean((y_rf_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_rf))

''' The results show that the random forest model performs slightly better than the
polynomial regression, in terms of RMSE and root mean square error. '''

# 7) GradientBoostingRegressor
model_gradientboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gradientboost_regressor', GradientBoostingRegressor(random_state=42))
])

model_gradientboost.fit(X_train, y_train)

y_gradientboost_pred = model_gradientboost.predict(X_test)

rmse_gradientboost = np.sqrt(np.mean((y_gradientboost_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_gradientboost))

mse_gradientboost = np.mean((y_gradientboost_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_gradientboost))

''' The results show that the GradientBoostingRegressor model does not perform as well as
like some other models tested (like LinearRegression and KNeighborsRegressor). '''

# 8) AdaBoostRegressor
model_adaboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('adaboost_regressor', AdaBoostRegressor())
])

model_adaboost.fit(X_train, y_train)

y_adaboost_pred = model_adaboost.predict(X_test)

rmse_adaboost = np.sqrt(np.mean((y_adaboost_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_adaboost))

mse_adaboost = np.mean((y_adaboost_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_adaboost))

''' Compared to other models trained on the same dataset, AdaBoostRegressor
performs worse than some models such as LinearRegression, PolynomialFeatures,
KNeighborsRegressor and XGBoost Regressor. AdaBoostRegressor also takes significantly 
more time for training than most other models.
    This model is not the best choice for this particular data set. '''

# 9) XGBoost Regressor
model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb_regressor', xgb.XGBRegressor())
])

model_xgb.fit(X_train, y_train)

y_xgb_pred = model_xgb.predict(X_test)

rmse_xgb = np.sqrt(np.mean((y_xgb_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_xgb))

mse_xgb = np.mean((y_xgb_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_xgb))

# A lower MSE value indicates a better fit of the model to the data.

# 10) BaggingRegressor
model_bagging = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('bagging_regressor', BaggingRegressor(estimator=xgb.XGBRegressor()))
])

model_bagging.fit(X_train, y_train)

y_bagging_pred = model_bagging.predict(X_test)

rmse_bagging = np.sqrt(np.mean((y_bagging_pred - y_test)**2))
print('RMSE: {:.2f}'.format(rmse_bagging))

mse_bagging = np.mean((y_bagging_pred - y_test)**2)
print('Mean Squared Error: {:.2f}'.format(mse_bagging))

''' The results show that the BaggingRegressor model performs slightly better than the
polynomial regression, in terms of RMSE and root mean square error. '''

# 11) BlendingRegressor
class BlendingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, estimators, meta_estimator):
        self.estimators = estimators
        self.meta_estimator = meta_estimator

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
        self.meta_estimator.fit(self._transform_base_preds(X), y)
        return self

    def predict(self, X):
        return self.meta_estimator.predict(self._transform_base_preds(X))

    def _transform_base_preds(self, X):
        base_preds = []
        for est in self.estimators:
            base_preds.append(est.predict(X))
        return np.column_stack(base_preds)


estimators = [
    LinearRegression(),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]

meta_estimator = LinearRegression()

model_blending = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('blending_regressor', BlendingRegressor(estimators, meta_estimator))
])

model_blending.fit(X_train, y_train)

y_blending_pred = model_blending.predict(X_test)

rmse_blending = np.sqrt(np.mean((y_blending_pred - y_test) ** 2))
print('RMSE: {:.2f}'.format(rmse_blending))

mse_blending = np.mean((y_blending_pred - y_test) ** 2)
print('Mean Squared Error: {:.2f}'.format(mse_blending))

''' Judging by the results, the XGBoost regressor showed the best results of all tested
models with the lowest RMSE and MSE values. '''

''' So, the models with the lowest RMSE and MSE values are:
    RandomForestRegressor, xgb.XGBRegressor (the best result), BaggingRegressor '''

# A) Cross Validation for RandomForestRegressor
cv_rf_regression = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores_rf_regression = cross_val_score(model_rf, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_rf_regression, scoring='neg_mean_squared_error')

print('RMSE: {:.2f}'.format(np.sqrt(-neg_mse_scores_rf_regression.mean())))

mse_scores_rf_regression = -neg_mse_scores_rf_regression
mse_mean_rf_regression = mse_scores_rf_regression.mean()

print('MSE: {:.2f}'.format(mse_mean_rf_regression))

# This cross-validation takes too much time.

# B) Cross Validation for xgb.XGBRegressor
cv_xgb_regression = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores_xgb_regression = cross_val_score(model_xgb, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_xgb_regression,
                                         scoring='neg_mean_squared_error')
mse_scores_xgb_regression = -neg_mse_scores_xgb_regression
mse_mean_xgb_regression = mse_scores_xgb_regression.mean()

print('RMSE: {:.2f}'.format(np.sqrt(mse_mean_xgb_regression)))
print('MSE: {:.2f}'.format(mse_mean_xgb_regression))

# Or
cv_xgb_regression = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores_xgb_regression = []
rmse_scores_xgb_regression = []

for train_index, test_index in cv_xgb_regression.split(analysis_data):
    X_train, X_test = analysis_data.drop(exclude_cols, axis=1).iloc[train_index], \
    analysis_data.drop(exclude_cols, axis=1).iloc[test_index]
    y_train, y_test = analysis_data['price'].iloc[train_index], analysis_data['price'].iloc[test_index]
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    mse_scores_xgb_regression.append(mean_squared_error(y_test, y_pred))
    rmse_scores_xgb_regression.append(np.sqrt(mean_squared_error(y_test, y_pred)))

mse_mean_xgb_regression = np.mean(mse_scores_xgb_regression)
rmse_mean_xgb_regression = np.sqrt(mse_mean_xgb_regression)

print('RMSE: {:.2f}'.format(rmse_mean_xgb_regression))
print('MSE: {:.2f}'.format(mse_mean_xgb_regression))

# C) Cross Validation for BaggingRegressor
cv_bagging = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores_bagging = cross_val_score(model_bagging, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_bagging, scoring='neg_mean_squared_error')
mse_scores_bagging = -neg_mse_scores_bagging
mse_mean_bagging = mse_scores_bagging.mean()

print('RMSE: {:.2f}'.format(np.sqrt(mse_mean_bagging)))
print('MSE: {:.2f}'.format(mse_mean_bagging))

''' The cross-validation results for the three models are as follows:
    RandomForestRegressor: RMSE = 613.24, MSE = 376062.17;
    XGBoost Regressor: RMSE = 581.91, MSE = 338620.85 (the best result);
    BaggingRegressor: RMSE = 593.78, MSE = 352580.60        
    All three models are evaluated using cross-validation with the KFold(5) strategy.
shuffling the data and using negative root mean square error as the evaluation metric.
    The xgb.XGBRegressor model has the best performance with the lowest RMS
error and root mean square error among the three models. '''

''' Overall, the MSE and RMSE values indicate that the models are not very good at 
predicting price. To improve the performance of the furniture price forecasting model, 
I recommend adding the following data into the source table:
    1) the cost of manufacturing products;
    2) the material from which the furniture is made;
    3) furniture rating (according to reviews on the website);
    4) warranty period of furniture;
    5) the number of photographs of furniture on the site; time of order placement;
    6) information about buyers: name (maybe the same person buys certain furniture),
gender, age, education, income level, residence: city / village, number of reviews on 
the site, engagement in social networks, online ordering time, duration of site viewing, 
online method order: website / social networks, from desktop device / from mobile device,
availability of email newsletters;
    7) time of purchase: time of year / month, working day / weekend / holiday, 
time of day;
    8) payment method: credit card / debit card / cash;
    9) cost, method and speed of delivery;
    10) whether there was a sale or not. '''
