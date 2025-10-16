from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import warnings

def MyModel(X, y, beta_true, method, quantile=0.5):
    """
    Fits a regression model based on the specified method and computes coefficient MSE.
    -------
    Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        beta_true : array-like, shape (n_features,)
            True coefficient vector for computing MSE.
        method : str
            The type of regression model to fit. Options are 'linear', 'quantile', 'huber'.
        quantile : float, optional
            The quantile to estimate if method is 'quantile'. Must be between 0 and
            1. Default is 0.5.
    -------
    Returns:
        results : dict
            A dictionary containing the model type, estimated coefficients, and MSE of beta.
    """


    
    if method == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        beta_hat = model.coef_
        
        # Compute MSE of coefficient estimates
        mse_beta = np.sum((beta_hat - beta_true) ** 2)
        
        results = {
            'model_type': 'Linear Regression',
            'beta_hat': beta_hat,
            'mse': mse_beta
        }
        return results
    
    if method == 'quantile':
        # Use robust solver for better convergence
        model = QuantileRegressor(
            quantile=quantile, 
            alpha=0,
            solver='highs'  # More robust solver
        )
        model.fit(X, y)
        beta_hat = model.coef_
        
        # Compute MSE of coefficient estimates
        mse_beta = np.sum((beta_hat - beta_true) ** 2)
        
        results = {
            'model_type': 'Quantile Regression',
            'beta_hat': beta_hat,
            'mse': mse_beta
        }
        return results

    if method == 'huber':
        # Standardize features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Increase max_iter and set tighter tolerance
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            model = HuberRegressor(
                max_iter=2000,
                epsilon=1.35,  # Default tuning parameter
                tol=1e-5,      # Convergence tolerance
                alpha=0.0001   # Small regularization can help convergence
            )
            model.fit(X_scaled, y)
        
        # Transform coefficients back to original scale
        beta_hat = model.coef_ / scaler.scale_
        
        # Compute MSE of coefficient estimates
        mse_beta = np.sum((beta_hat - beta_true) ** 2)
        
        results = {
            'model_type': 'Huber Regression',
            'beta_hat': beta_hat,
            'mse': mse_beta
        }
        return results
    
    if method == 'ridgeless':
        n, p = X.shape
        
        if p < n:
            # Underparameterized case: β̂ = (X'X)^(-1)X'y
            beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
        else:
            # Overparameterized case: β̂ = X'(XX')^(-1)y
            beta_hat = X.T @ np.linalg.solve(X @ X.T, y)
        
        # Compute MSE of coefficient estimates
        mse_beta = np.sum((beta_hat - beta_true) ** 2)
        
        results = {
            'model_type': 'Ridgeless Regression',
            'beta_hat': beta_hat,
            'mse': mse_beta
        }
        return results
    
    raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'quantile', 'huber', or 'ridgeless'.")
