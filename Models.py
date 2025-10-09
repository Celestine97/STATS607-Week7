from sklearn import linear_model.LinearRegression
from sklearn import linear_model.QuantileRegressor
from sklearn import linear_model.HuberRegressor
import pandas as pd

def MyModel(X, y, method, quantile=None):
    if method == 'linear':
        model = linear_model.LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        results = pd.DataFrame({
            'model_type': 'Linear Regression',
            'mse': ((y - y_pred) ** 2).mean()
        })
        return results
    
    if method == 'quantile':
        model = linear_model.QuantileRegressor(alpha=quantile)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        results = pd.DataFrame({
            'model_type': 'Quantile Regression',
            'mse': ((y - y_pred) ** 2).mean()
        })
        return results

    if method == 'huber':
        model = linear_model.HuberRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        results = pd.DataFrame({
            'model_type': 'Huber Regression',
            'mse': ((y - y_pred) ** 2).mean()
        })
        return results
