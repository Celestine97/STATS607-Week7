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


def generate_linear_model(n=200, gamma=1.0, sigma2=1.0, r2=5.0, rng=None):
    """
    Generate data for the linear model:
        y = Xβ + σε,   X_ij, ε_i ~ N(0,1)
        β = sqrt(r² / p) * 1_p
    where p = floor(gamma * n).

    Parameters
    ----------
    n : int
        Number of observations.
    gamma : float
        Aspect ratio (defines p = floor(gamma * n)).
    sigma2 : float
        Noise variance.
    r2 : float
        Signal strength parameter.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Response vector.
    beta : np.ndarray of shape (p,)
        True coefficient vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = int(np.floor(gamma * n))
    sigma = np.sqrt(sigma2)

    # Generate X and epsilon
    X = rng.normal(0, 1, size=(n, p))
    eps = rng.normal(0, 1, size=n)

    # Define beta
    beta = np.sqrt(r2 / p) * np.ones(p)

    # Generate response
    y = X @ beta + sigma * eps

    return X, y, beta


# Example usage
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    sigma2 = 1
    r2 = 5

    for gamma in [0.1, 0.5, 1, 2, 5, 10]:
        X, y, beta = generate_linear_model(n=n, gamma=gamma, sigma2=sigma2, r2=r2, rng=rng)
        print(f"γ={gamma:.1f}: X.shape={X.shape}, ||β||²={np.sum(beta**2):.3f}, y.var={np.var(y):.3f}")


