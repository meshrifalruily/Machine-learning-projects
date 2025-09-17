from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib

dataset = fetch_california_housing()
print(dataset)

data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['target'] = dataset.target

#print(type(dataset))
X = data.drop('target', axis=1)
y = data['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)    

model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('sgd', SGDRegressor())     
])
     

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {np.sqrt(mse)}')

# Grid Search for hyperparameter tuning
param_grid = {
    'imputer__strategy': ['mean', 'median'],
    'sgd__max_iter': [1000, 5000, 10000], 
    'sgd__eta0': [0.001, 0.01, 0.1] 
    }

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f'Best parameters from Grid Search: {grid_search.best_params_}')
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print(f'Root Mean Squared Error after Grid Search: {np.sqrt(mse_best)}') 
joblib.dump(best_model, 'best_model.joblib')   

model = joblib.load('best_model.joblib')
y_pred_loaded = model.predict(X_test)
mse_loaded = mean_squared_error(y_test, y_pred_loaded)
print(f'Root Mean Squared Error from loaded model: {np.sqrt(mse_loaded)}')
