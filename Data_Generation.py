import numpy as np
from scipy import stats

def generate_design_matrix(n, p, rho=0.5, rng=None):
    """
    Generate design matrix X with autoregressive correlation structure.
    
    Parameters:
    -----------
    n : int
        Number of observations
    p : int
        Number of predictors
    rho : float
        Autoregressive correlation parameter (0 <= rho < 1)
    rng : np.random.Generator, optional
        Random number generator
    
    Returns:
    --------
    X : np.ndarray of shape (n, p)
        Design matrix
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate covariance matrix with autoregressive structure: Cov(X_j, X_k) = rho^|j-k|
    cov_matrix = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            cov_matrix[j, k] = rho ** np.abs(j - k)
    
    # Generate multivariate normal data
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)
    
    return X

def generate_beta(p, snr, X, sigma=1.0, rng=None):
    """
    Generate coefficient vector β scaled to achieve target SNR.
    
    Parameters:
    -----------
    p : int
        Number of predictors
    snr : float
        Target signal-to-noise ratio
    X : np.ndarray of shape (n, p)
        Design matrix
    sigma : float
        Error standard deviation (for t-dist with df>2, this is approx. the scale)
    rng : np.random.Generator, optional
        Random number generator
    
    Returns:
    --------
    beta : np.ndarray of shape (p,)
        Coefficient vector
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random coefficients
    beta = rng.standard_normal(p)
    
    # Scale to achieve target SNR
    # SNR = β'X'Xβ / σ²
    XtX = X.T @ X
    current_signal = beta.T @ XtX @ beta
    target_signal = snr * (sigma ** 2)
    
    scaling_factor = np.sqrt(target_signal / current_signal)
    beta = beta * scaling_factor
    
    return beta

def generate_errors(n, df, rng=None):
    """
    Generate error terms from t-distribution.
    
    Parameters:
    -----------
    n : int
        Number of observations
    df : float
        Degrees of freedom for t-distribution
        Use df=np.inf for normal distribution
    rng : np.random.Generator, optional
        Random number generator
    
    Returns:
    --------
    errors : np.ndarray of shape (n,)
        Error terms
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if df == np.inf:
        # Normal distribution
        errors = rng.standard_normal(n)
    else:
        # Student-t distribution
        errors = stats.t.rvs(df=df, size=n, random_state=rng)
        # Standardize to have unit variance (for df > 2)
        if df > 2:
            errors = errors / np.sqrt(df / (df - 2))
    
    return errors

def generate_data(n, p, beta, df, rho=0.5, rng=None):
    """
    Generate complete dataset: y = Xβ + ε
    
    Parameters:
    -----------
    n : int
        Number of observations
    p : int
        Number of predictors
    beta : np.ndarray
        True coefficient vector
    df : float
        Degrees of freedom for error distribution
        (Use df=np.inf for normal errors)
    rho : float
        AR correlation parameter
    rng : np.random.Generator, optional
        Random number generator
    
    Returns:
    --------
    X : np.ndarray of shape (n, p)
        Design matrix
    y : np.ndarray of shape (n,)
        Response vector
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate design matrix
    X = generate_design_matrix(n, p, rho=rho, rng=rng)
    
    # Generate errors
    errors = generate_errors(n, df=df, rng=rng)
    
    # Generate response
    y = X @ beta + errors
    
    return X, y

# Example usage:
if __name__ == "__main__":
    # Set ONE master seed at the top
    seed = 42
    rng = np.random.default_rng(seed)
    
    # Set parameters
    n = 100
    gamma = 0.5  # aspect ratio
    p = int(gamma * n)
    snr = 5
    df = 3
    rho = 0.5
    
    # Generate data - pass the same rng to all functions
    X_temp = generate_design_matrix(n, p, rho=rho, rng=rng)
    beta = generate_beta(p, snr=snr, X=X_temp, rng=rng)
    X, y = generate_data(n, p, beta=beta, df=df, rho=rho, rng=rng)
    
    print(f"Generated data: X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"True beta (first 5): {beta[:5]}")