from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import HuberRegressor
import pandas as pd

def MyModel(X, y, method, quantile=None):
    """
    Fits a regression model based on the specified method.
    -------
    Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        method : str
            The type of regression model to fit. Options are 'linear', 'quantile', 'huber'.
        quantile : float, optional
            The quantile to estimate if method is 'quantile'. Must be between 0 and
            1. Default is None.
    -------
    Returns:
        results : pd.DataFrame
            A DataFrame containing the model type and mean squared error.
    """
    
    if method == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        results = {
            'model_type': 'Linear Regression',
            'mse': ((y - y_pred) ** 2).mean()
        }
        return results
    
    if method == 'quantile':
        model = QuantileRegressor(quantile=quantile, alpha=0)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        results = {
            'model_type': 'Quantile Regression',
            'mse': ((y - y_pred) ** 2).mean()
        }
        return results

    if method == 'huber':
        model = HuberRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        results = {
            'model_type': 'Huber Regression',
            'mse': ((y - y_pred) ** 2).mean()
        }
        return results
