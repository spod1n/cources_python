import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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

r2_knn = r2_score(y_test, y_knn_pred)

''' Let's evaluate the performance of the model:
    RMSE is the square root of MSE (mean square error) and is more interpretable
value because it is in the same units as the forecast and actual values. '''

rmse_knn = np.sqrt(np.mean((y_knn_pred - y_test)**2))
print('RMSE of KNeighborsRegressor: {:.2f}'.format(rmse_knn))

mse_knn = np.mean((y_knn_pred - y_test)**2)
print('Mean Squared Error of KNeighborsRegressor: {:.2f}'.format(mse_knn))

print(f'r2_score of KNeighborsRegressor is {r2_knn:.3f}')

# 2) LinearRegression
model_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_regression.fit(X_train, y_train)

y_regression_pred = model_regression.predict(X_test)

r2_regression = r2_score(y_test, y_regression_pred)

rmse_regression = np.sqrt(np.mean((y_regression_pred - y_test)**2))
print('RMSE of LinearRegression: {:.2f}'.format(rmse_regression))

mse_regression = np.mean((y_regression_pred - y_test)**2)
print('Mean Squared Error of LinearRegression: {:.2f}'.format(mse_regression))

print(f'r2_score of LinearRegression is {r2_regression:.3f}')

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

r2_tree_regression = r2_score(y_test, y_tree_regression_pred)

rmse_tree_regression = np.sqrt(np.mean((y_tree_regression_pred - y_test)**2))
print('RMSE of DecisionTreeRegressor: {:.2f}'.format(rmse_tree_regression))

mse_tree_regression = np.mean((y_tree_regression_pred - y_test)**2)
print('Mean Squared Error of DecisionTreeRegressor: {:.2f}'.format(mse_tree_regression))

print(f'r2_score of DecisionTreeRegressor is {r2_tree_regression:.3f}')

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

print('Best hyperparameters of DecisionTreeRegressor:', grid_search.best_params_)
print('Best score of DecisionTreeRegressor:', grid_search.best_score_)

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

r2_best_tree_regression = r2_score(y_test, y_best_tree_regression_pred)

rmse_best_tree_regression = np.sqrt(np.mean((y_best_tree_regression_pred - y_test)**2))
print('RMSE of DecisionTreeRegressor with hyperparameters: {:.2f}'.format(rmse_best_tree_regression))

mse_best_tree_regression = np.mean((y_best_tree_regression_pred - y_test)**2)
print('Mean Squared Error of DecisionTreeRegressor with hyperparameters: {:.2f}'.format(mse_best_tree_regression))

print(f'r2_score of DecisionTreeRegressor with hyperparameters is {r2_best_tree_regression:.3f}')

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

r2_elasticnet = r2_score(y_test, y_elasticnet_pred)

rmse_elasticnet = np.sqrt(np.mean((y_elasticnet_pred - y_test)**2))
print('RMSE of ElasticNet: {:.2f}'.format(rmse_elasticnet))

mse_elasticnet = np.mean((y_elasticnet_pred - y_test)**2)
print('Mean Squared Error of ElasticNet: {:.2f}'.format(mse_elasticnet))

print(f'r2_score of ElasticNet is {r2_elasticnet:.3f}')

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

print('Best hyperparameters of ElasticNet:', grid_search.best_params_)
print('Best score of ElasticNet:', grid_search.best_score_)

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

r2_best_elasticnet = r2_score(y_test, y_best_elasticnet_pred)

rmse_best_elasticnet = np.sqrt(np.mean((y_best_elasticnet_pred - y_test)**2))
print('RMSE of ElasticNet with hyperparameters: {:.2f}'.format(rmse_best_elasticnet))

mse_best_elasticnet = np.mean((y_best_elasticnet_pred - y_test)**2)
print('Mean Squared Error of ElasticNet with hyperparameters: {:.2f}'.format(mse_best_elasticnet))

print(f'r2_score of ElasticNet with hyperparameters is {r2_best_elasticnet:.3f}')

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

r2_polynomial = r2_score(y_test, y_polynomial_pred)

rmse_polynomial = np.sqrt(np.mean((y_polynomial_pred - y_test)**2))
print('RMSE of PolynomialFeatures: {:.2f}'.format(rmse_polynomial))

mse_polynomial = np.mean((y_polynomial_pred - y_test)**2)
print('Mean Squared Error of PolynomialFeatures: {:.2f}'.format(mse_polynomial))

print(f'r2_score of PolynomialFeatures is {r2_polynomial:.3f}')

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

r2_rf = r2_score(y_test, y_rf_pred)

rmse_rf = np.sqrt(np.mean((y_rf_pred - y_test)**2))
print('RMSE of RandomForestRegressor: {:.2f}'.format(rmse_rf))

mse_rf = np.mean((y_rf_pred - y_test)**2)
print('Mean Squared Error of RandomForestRegressor: {:.2f}'.format(mse_rf))

print(f'r2_score of RandomForestRegressor is {r2_rf:.3f}')

''' The results show that the random forest model performs slightly better than the
polynomial regression, in terms of RMSE and root mean square error. '''

# 7) GradientBoostingRegressor
model_gradientboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gradientboost_regressor', GradientBoostingRegressor(random_state=42))
])

model_gradientboost.fit(X_train, y_train)

y_gradientboost_pred = model_gradientboost.predict(X_test)

r2_gradientboost = r2_score(y_test, y_gradientboost_pred)

rmse_gradientboost = np.sqrt(np.mean((y_gradientboost_pred - y_test)**2))
print('RMSE of GradientBoostingRegressor: {:.2f}'.format(rmse_gradientboost))

mse_gradientboost = np.mean((y_gradientboost_pred - y_test)**2)
print('Mean Squared Error of GradientBoostingRegressor: {:.2f}'.format(mse_gradientboost))

print(f'r2_score of GradientBoostingRegressor is {r2_gradientboost:.3f}')

''' The results show that the GradientBoostingRegressor model does not perform as well as
like some other models tested (like LinearRegression and KNeighborsRegressor). '''

# 8) AdaBoostRegressor
model_adaboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('adaboost_regressor', AdaBoostRegressor())
])

model_adaboost.fit(X_train, y_train)

y_adaboost_pred = model_adaboost.predict(X_test)

r2_adaboost = r2_score(y_test, y_adaboost_pred)

rmse_adaboost = np.sqrt(np.mean((y_adaboost_pred - y_test)**2))
print('RMSE of AdaBoostRegressor: {:.2f}'.format(rmse_adaboost))

mse_adaboost = np.mean((y_adaboost_pred - y_test)**2)
print('Mean Squared Error of AdaBoostRegressor: {:.2f}'.format(mse_adaboost))

print(f'r2_score of AdaBoostRegressor is {r2_adaboost:.3f}')

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

r2_xgb = r2_score(y_test, y_xgb_pred)

rmse_xgb = np.sqrt(np.mean((y_xgb_pred - y_test)**2))
print('RMSE of XGBoost Regressor: {:.2f}'.format(rmse_xgb))

mse_xgb = np.mean((y_xgb_pred - y_test)**2)
print('Mean Squared Error of XGBoost Regressor: {:.2f}'.format(mse_xgb))

print(f'r2_score of XGBoost Regressor is {r2_xgb:.3f}')

# A lower MSE value indicates a better fit of the model to the data.

# 10) BaggingRegressor
model_bagging = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('bagging_regressor', BaggingRegressor(estimator=xgb.XGBRegressor()))
])

model_bagging.fit(X_train, y_train)

y_bagging_pred = model_bagging.predict(X_test)

r2_bagging = r2_score(y_test, y_bagging_pred)

rmse_bagging = np.sqrt(np.mean((y_bagging_pred - y_test)**2))
print('RMSE of BaggingRegressor: {:.2f}'.format(rmse_bagging))

mse_bagging = np.mean((y_bagging_pred - y_test)**2)
print('Mean Squared Error of BaggingRegressor: {:.2f}'.format(mse_bagging))

print(f'r2_score of BaggingRegressor is {r2_bagging:.3f}')

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

r2_blending = r2_score(y_test, y_blending_pred)

rmse_blending = np.sqrt(np.mean((y_blending_pred - y_test) ** 2))
print('RMSE of BlendingRegressor: {:.2f}'.format(rmse_blending))

mse_blending = np.mean((y_blending_pred - y_test) ** 2)
print('Mean Squared Error of BlendingRegressor: {:.2f}'.format(mse_blending))

print(f'r2_score of BlendingRegressor is {r2_blending:.3f}')

''' Judging by the results, the XGBoost regressor showed the best results of all tested
models with the lowest RMSE and MSE values. '''

''' So, the models with the lowest RMSE and MSE values are:
    RandomForestRegressor, xgb.XGBRegressor (the best result), BaggingRegressor '''

# A) Cross Validation for RandomForestRegressor
cv_rf_regression = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores_rf_regression = cross_val_score(model_rf, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_rf_regression, scoring='neg_mean_squared_error')

print('RMSE of Cross Validation for RandomForestRegressor: {:.2f}'.format(np.sqrt(-neg_mse_scores_rf_regression.mean())))

mse_scores_rf_regression = -neg_mse_scores_rf_regression
mse_mean_rf_regression = mse_scores_rf_regression.mean()

print('MSE of Cross Validation for RandomForestRegressor: {:.2f}'.format(mse_mean_rf_regression))

r2_rf_regression = r2_score(analysis_data['price'], model_rf.predict(analysis_data.drop(exclude_cols, axis=1)))
print(f'r2_score of Cross Validation for RandomForestRegressor is {r2_rf_regression:.3f}')

# This cross-validation takes too much time.

# B) Cross Validation for xgb.XGBRegressor
cv_xgb_regression = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores_xgb_regression = cross_val_score(model_xgb, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_xgb_regression,
                                         scoring='neg_mean_squared_error')
mse_scores_xgb_regression = -neg_mse_scores_xgb_regression
mse_mean_xgb_regression = mse_scores_xgb_regression.mean()

print('RMSE of Cross Validation for xgb.XGBRegressor: {:.2f}'.format(np.sqrt(mse_mean_xgb_regression)))
print('MSE of Cross Validation for xgb.XGBRegressor: {:.2f}'.format(mse_mean_xgb_regression))

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

print('RMSE of Cross Validation for xgb.XGBRegressor: {:.2f}'.format(rmse_mean_xgb_regression))
print('MSE of Cross Validation for xgb.XGBRegressor: {:.2f}'.format(mse_mean_xgb_regression))

r2_xgb_regression = r2_score(analysis_data['price'], model_xgb.predict(analysis_data.drop(exclude_cols, axis=1)))
print(f"r2_score of Cross Validation for xgb.XGBRegressor: {r2_xgb_regression:.2f}")

# C) Cross Validation for BaggingRegressor
cv_bagging = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores_bagging = cross_val_score(model_bagging, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_bagging, scoring='neg_mean_squared_error')
mse_scores_bagging = -neg_mse_scores_bagging
mse_mean_bagging = mse_scores_bagging.mean()

print('RMSE of Cross Validation for xgb.BaggingRegressor: {:.2f}'.format(np.sqrt(mse_mean_bagging)))
print('MSE of Cross Validation for xgb.BaggingRegressor: {:.2f}'.format(mse_mean_bagging))

model_bagging.fit(X_train, y_train)
y_bagging_pred = model_bagging.predict(X_test)
r2_bagging = r2_score(y_test, y_bagging_pred)
print(f'R2 Score of BaggingRegressor on Test Set: {r2_bagging:.3f}')

'''
r2_score (Коефіцієнт детермінації):
У контексті регресійних моделей r2_score є корисним показником для оцінки відповідності. 
r2_score є статистичним показником того, наскільки прогнозовані значення відповідають фактичним значенням. 
Інтерпретація:  вимірює частку дисперсії залежної змінної, яку можна передбачити на основі незалежних змінних.
Мета: Вища  r2_score вказує на кращу пояснювальну силу. 
Він коливається від 0 до 1, де 1 вказує на ідеальне передбачення, а значення нижче 0 вказують на те, 
що модель не краща, ніж просто передбачення середнього значення залежної змінної, тобто модель не пояснює 
жодної мінливості цільової змінної.

RMSE (середньоквадратична помилка):
Це загальновживаний показник для оцінки точності прогнозної моделі, зокрема в контексті регресійного аналізу.
Інтерпретація: RMSE вимірює середню величину похибок між прогнозованими та фактичними значеннями.
Вона виражається в тих самих одиницях, що й цільова змінна.
Мета: нижчий RMSE вказує на кращу точність прогнозування. Ви хочете мінімізувати RMSE.
 
Вищий r2_score вказує на кращу відповідність моделі даним, тоді як нижча RMSE вказує на кращу точність 
прогнозу. 
r2_score  (коефіцієнт детермінації) і RMSE (середньоквадратична помилка) дійсно пов’язані, але вони 
вимірюють різні аспекти продуктивності моделі.    
На r2_score можуть впливати різні фактори, і він не завжди може ідеально узгоджуватися з іншими показниками, 
такими як RMSE.  
   
Що важливіше?

Для передбачення:
Якщо вашою основною метою є точне передбачення, більш важливим є нижчий RMSE. 
Ви хочете, щоб ваша модель робила прогнози, близькі до справжніх значень.

Для пояснення:
Якщо ваша мета — зрозуміти взаємозв’язок між змінними, а прогноз — другорядна проблема, r2_score  
може бути важливішим. Він говорить вам, наскільки добре ваші незалежні змінні пояснюють мінливість 
залежної змінної.

Балансування:
У багатьох випадках хороша модель повинна знайти баланс між точністю передбачення та пояснювальною силою.
'''
