import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from analysis import plot_mse_vs_df, plot_small_multiples
from Models import MyModel
from Data_Generation import *


def evaluate_one_run(n, ar, df, rho, snr, rep, methods, rng):
    """
    Run one simulation replicate and evaluate multiple methods on the same dataset.
    Returns a list of result dicts (one per method).

    n: number of samples, fixed=200 here
    ar: aspect ratio (p/n), here is gamma in the setting
    """
    p = int(n * ar)

    # Generate data with true beta
    # X_temp = generate_design_matrix(n, p, rho=rho, rng=rng)
    # beta = generate_beta(p, snr=snr, X=X_temp, rng=rng)
    # X, y = generate_data(n, p, beta=beta, df=df, rho=rho, rng=rng)
    X, y, beta = generate_linear_model(n=n, gamma=ar, r2=snr, sigma2=1.0, rng=rng)

    results = []
    methods = ['ridgeless']  # Currently only ridgeless implemented

    model_results = MyModel(X, y, beta_true=beta, method=methods[0], quantile=0.5)
    mse = model_results["mse"]

    results.append({
        "method": methods[0],
        "n": n,
        "ar": ar,
        "df": df,
        "rho": rho,
        "SNR": snr,
        "rep": rep,
        "mse": mse,
    })
    return results

def run_simulations(
    n=200,
    aspect_ratio=[0.2, 0.5, 0.8],
    dfs=[1, 2, 3, 5, 10, 15, 20, np.inf],
    rhos=[0.2],
    snrs=[1, 5, 10],
    reps=[1, 50, 1000],
    methods=['ridgeless'],
    n_jobs=-1,
    seed=0,
):
    """
    Run simulation grid: for each combination of parameters,
    Run 5000 times (rep=1), 100 times (rep=50), or 5 times (rep=1000).
    Computes MSE of coefficient estimates: MSE(β̂) = mean((β̂ - β)²)
    """

    rng_master = np.random.default_rng(seed)
    for rep in reps:
        if rep not in [1, 50, 1000]:
            raise ValueError("rep must be one of [1, 50, 1000]")

        if rep == 1:
            gamma_values = np.logspace(np.log10(0.1), np.log10(10), 5000)
        elif rep == 50:
            gamma_values = np.logspace(np.log10(0.1), np.log10(10), 100)
        elif rep == 1000:
            gamma_values = np.array([0.2, 0.5, 0.8, 2, 5])
        param_grid = [
            (ar, rep)
            for ar in gamma_values
            for rep in range(reps)
        ]

        print(f"Running {len(param_grid)} total simulation runs...")


    results_nested = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_one_run)(
            ar, rep,
            methods=methods,
        rng=np.random.default_rng(rng_master.integers(1e9))
        )
        for (ar, rep) in param_grid
    )
    results = [res for sublist in results_nested for res in sublist]

    df_results = pd.DataFrame(results)
    df_results.to_csv("simulation_results.csv", index=False)
    print(f"Saved results to simulation_results.csv with {len(df_results)} rows.")

    return df_results

def analyze_results(df_results):
    """
    Aggregate results by df and method, then call plotting functions.
    """
    df_summary = (
        df_results.groupby(["SNR", "df", "method"])["mse"]
        .mean()
        .reset_index()
    )

    plot_mse_vs_df(df_summary, output="mse_vs_df.png")
    plot_small_multiples(df_results, output="mse_small_multiples.png")

if __name__ == "__main__":
    df_results = run_simulations()
    analyze_results(df_results)
    print("Simulation and analysis complete.")