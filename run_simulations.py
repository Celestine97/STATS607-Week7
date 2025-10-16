import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from analysis import plot_figure2
from Models import MyModel
from Data_Generation import generate_linear_model


def evaluate_one_run(n, gamma, r2, sigma2, rep, rng):
    """
    Run one simulation replicate for ridgeless regression.
    
    Parameters:
    -----------
    n : int
        Number of observations (fixed at 200)
    gamma : float
        Aspect ratio p/n
    r2 : float
        Signal strength (fixed at 5)
    sigma2 : float
        Noise variance (fixed at 1)
    rep : int
        Replicate number
    rng : np.random.Generator
        Random number generator
    
    Returns:
    --------
    dict with gamma, rep, and MSE
    """
    # Generate data
    X, y, beta = generate_linear_model(n=n, gamma=gamma, r2=r2, sigma2=sigma2, rng=rng)
    
    # Fit ridgeless regression
    model_results = MyModel(X, y, beta_true=beta, method='ridgeless')
    mse = model_results["mse"]
    
    return {
        "gamma": gamma,
        "rep": rep,
        "mse": mse,
    }


def run_simulations(
    n=200,
    r2=5.0,
    sigma2=1.0,
    n_jobs=-1,
    seed=0,
):
    """
    Run simulation for ridgeless regression across three regimes.
    
    Regime 1 (nsim=1): 5000 scenarios, gamma in [0.1, 10] (log-spaced)
    Regime 2 (nsim=50): 100 scenarios, gamma in [0.1, 10] (log-spaced)
    Regime 3 (nsim=1000): 5 scenarios, gamma in {0.2, 0.5, 0.8, 2, 5}
    
    For each scenario, compute:
    - MSE_hat: mean of MSE across replicates
    - se_MSE: standard error of MSE (0 when nsim=1)
    """
    rng_master = np.random.default_rng(seed)
    
    # Define the three regimes
    regimes = [
        (1, np.logspace(np.log10(0.1), np.log10(10), 5000)),
        (50, np.logspace(np.log10(0.1), np.log10(10), 100)),
        (1000, np.array([0.2, 0.5, 0.8, 2, 5]))
    ]
    
    for nsim, gamma_values in regimes:
        print(f"\n{'='*60}")
        print(f"Running regime: nsim={nsim}, n_scenarios={len(gamma_values)}")
        print(f"{'='*60}")
        
        # Create parameter grid: (gamma, rep)
        param_grid = [
            (gamma, rep)
            for gamma in gamma_values
            for rep in range(nsim)
        ]
        
        total_runs = len(param_grid)
        print(f"Total simulation runs: {total_runs}")
        
        # Run simulations in parallel
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(evaluate_one_run)(
                n=n,
                gamma=gamma,
                r2=r2,
                sigma2=sigma2,
                rep=rep,
                rng=np.random.default_rng(rng_master.integers(1e9))
            )
            for (gamma, rep) in param_grid
        )
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Aggregate by gamma: compute mean and SE
        if nsim == 1:
            # No SE when only 1 replicate
            df_summary = df_results.groupby('gamma').agg(
                MSE_hat=('mse', 'mean'),
            ).reset_index()
            df_summary['se_MSE'] = 0.0
            df_summary['nsim'] = 1
        else:
            # Compute mean and SE across replicates
            df_summary = df_results.groupby('gamma').agg(
                MSE_hat=('mse', 'mean'),
                se_MSE=('mse', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
            ).reset_index()
            df_summary['nsim'] = nsim
        
        # Reorder columns
        df_summary = df_summary[['nsim', 'gamma', 'MSE_hat', 'se_MSE']]
        
        # Save both formats
        csv_file = f"simulation_results_nsim{nsim}.csv"
        pkl_file = f"simulation_results_nsim{nsim}.pkl"
        
        df_summary.to_csv(csv_file, index=False)
        df_summary.to_pickle(pkl_file)
        
        print(f"\n✓ Saved {len(df_summary)} scenarios to:")
        print(f"  - {csv_file}")
        print(f"  - {pkl_file}")
        
        # Print summary statistics
        print(f"\nSummary statistics:")
        print(f"  Gamma range: [{df_summary['gamma'].min():.3f}, {df_summary['gamma'].max():.3f}]")
        print(f"  MSE range: [{df_summary['MSE_hat'].min():.3f}, {df_summary['MSE_hat'].max():.3f}]")
        if nsim > 1:
            print(f"  SE range: [{df_summary['se_MSE'].min():.4f}, {df_summary['se_MSE'].max():.4f}]")


if __name__ == "__main__":    
    print("Starting ridgeless regression simulation study...")
    print("Parameters: n=200, r²=5, σ²=1")
    print("\nThis will run 3 regimes:")
    print("  - nsim=1: 5000 scenarios")
    print("  - nsim=50: 100 scenarios")
    print("  - nsim=1000: 5 scenarios")
    print("\nTotal runs: 5000 + 5000 + 5000 = 15,000")
    print("="*60)
    
    run_simulations()
    
    print("\n" + "="*60)
    print("✓ Simulation complete!")
    print("✓ Files generated:")
    print("  - simulation_results_nsim1.csv")
    print("  - simulation_results_nsim50.csv")
    print("  - simulation_results_nsim1000.csv")
    print("\nNext step: Run analysis to generate Figure 2")
    print("="*60)