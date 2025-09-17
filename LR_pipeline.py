from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error  
dataset = fetch_california_housing()
print(dataset)

data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['target'] = dataset.target

#print(type(dataset))
X = data.drop('target', axis=1)
y = data['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)    

model = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=10000, eta0=0.01)
)       

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {np.sqrt(mse)}')

