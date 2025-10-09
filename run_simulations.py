import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from analysis import plot_mse_vs_df, plot_small_multiples
from Models import  MyModel
from Data_Generation import *
!pip install joblib
def evaluate_one_run(n, ar, df, rho, snr, rep, method, rng):
    """
    Run one simulation replicate and return a result dict.
    """
    p = int(n* ar)
    X_temp = generate_design_matrix(n, p, rho=rho, rng=rng)
    beta = generate_beta(p, snr=snr, X=X_temp, rng=rng)
    X, y = generate_data(n, p, beta=beta, df=df, rho=rho, rng=rng)
    model_results = MyModel(X, y, method, quantile=0.5)
    mse = model_results["mse"]

    return {
        "method": method,
        "n": n,
        "ar": ar,
        "df": df,
        "rho": rho,
        "SNR": snr,
        "rep": rep,
        "mse": mse,
    }


def run_simulations(n=200,aspect_ratio = [0.2,0.5,0.8],dfs=[1, 2, 3, 20, np.inf],rhos=[0.2],snrs=[1, 5, 10],reps=30,methods=["linear", "quantile", "huber"],n_jobs=-1,seed=42,):
    rng_master = np.random.default_rng(seed)
    param_grid = [
        (ar, df, rho, snr, rep, method)
        for ar in aspect_ratio
        for df in dfs
        for rho in rhos
        for snr in snrs
        for rep in range(reps)
        for method in methods
    ]

    print(f"Running {len(param_grid)} total simulation runs...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_one_run)(n, ar, df, rho, snr, rep, method,rng=np.random.default_rng(rng_master.integers(1e9)))
        for (ar,df, rho, snr, rep, method) in param_grid
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv("simulation_results.csv", index=False)
    print("Saved results to simulation_results.csv")

    return df_results

def analyze_results(df_results):
    """
    Aggregate results by df and method, then call plotting functions.
    """
    df_summary = (
        df_results.groupby(["df", "method"])["mse"]
        .mean()
        .reset_index()
        .sort_values(["method", "df"])
    )

    plot_mse_vs_df(df_summary, output="mse_vs_df.png")
    plot_small_multiples(df_results, output="mse_small_multiples.png")

if __name__ == "__main__":
    df_results = run_simulations()
    analyze_results(df_results)
    print("Simulation and analysis complete.")
