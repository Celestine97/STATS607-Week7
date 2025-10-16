import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_figure2(output="figure2_reproduction.png"):
    """
    Reproduce Figure 2 from Hastie et al. (2022):
    Three subplots (one per nsim regime) showing MSE vs gamma.
    
    - For nsim=1: scatter plot with LOWESS smoothing
    - For nsim=50, 1000: scatter + error bars (±2*SE)
    - All plots show theoretical risk curve and vertical line at gamma=1
    """
    
    # Load all three regimes
    regimes = [1, 50, 1000]
    dfs = {}
    
    for nsim in regimes:
        pkl_file = f"simulation_results_nsim{nsim}.pkl"
        try:
            dfs[nsim] = pd.read_pickle(pkl_file)
        except FileNotFoundError:
            print(f"Warning: {pkl_file} not found. Run simulations first.")
            return
    
    # Theoretical risk (from paper)
    # For gamma < 1: R = sigma^2 * gamma / (1 - gamma)
    # For gamma > 1: R = r^2 * (1 - 1/gamma) + sigma^2 / (gamma - 1)
    def theoretical_risk(gamma, r2=5, sigma2=1):
        risk = np.zeros_like(gamma)
        mask_under = gamma < 1
        mask_over = gamma > 1
        
        risk[mask_under] = sigma2 * gamma[mask_under] / (1 - gamma[mask_under])
        risk[mask_over] = r2 * (1 - 1/gamma[mask_over]) + sigma2 / (gamma[mask_over] - 1)
        
        return risk
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, nsim in enumerate(regimes):
        ax = axes[idx]
        df = dfs[nsim]
        
        # Plot theoretical curve
        gamma_theory = np.concatenate([
            np.linspace(0.1, 0.99, 200),
            np.linspace(1.01, 10, 200)
        ])
        risk_theory = theoretical_risk(gamma_theory)
        ax.plot(gamma_theory, risk_theory, 'k-', linewidth=2, label='Theoretical', zorder=1)
        
        # Vertical line at gamma=1
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        
        if nsim == 1:
            # Scatter with low alpha + LOWESS smoothing
            ax.scatter(df['gamma'], df['MSE_hat'], alpha=0.2, s=10, color='blue', zorder=2)
            
            # LOWESS smoothing (separate for gamma < 1 and gamma > 1)
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Under-parameterized
            df_under = df[df['gamma'] <= 1].sort_values('gamma')
            if len(df_under) > 10:
                smoothed_under = lowess(df_under['MSE_hat'], df_under['gamma'], frac=0.1)
                ax.plot(smoothed_under[:, 0], smoothed_under[:, 1], 'b-', linewidth=2, 
                       label='LOWESS (γ≤1)', zorder=3)
            
            # Over-parameterized
            df_over = df[df['gamma'] >= 1].sort_values('gamma')
            if len(df_over) > 10:
                smoothed_over = lowess(df_over['MSE_hat'], df_over['gamma'], frac=0.1)
                ax.plot(smoothed_over[:, 0], smoothed_over[:, 1], 'r-', linewidth=2,
                       label='LOWESS (γ≥1)', zorder=3)
        
        else:
            # Scatter + error bars
            ax.errorbar(df['gamma'], df['MSE_hat'], 
                       yerr=2*df['se_MSE'],  # ±2*SE
                       fmt='o', markersize=6, capsize=4, capthick=1.5,
                       color='blue', ecolor='lightblue', alpha=0.8,
                       label='Estimated', zorder=2)
        
        # Formatting
        ax.set_xlabel('γ = p/n', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'nsim = {nsim}', fontsize=12, fontweight='bold')
        
        # X-axis: split scale (0.1-1 takes half, 1-10 takes half)
        # This is tricky - for now use log scale
        ax.set_xscale('log')
        ax.set_xlim(0.09, 11)
        ax.set_ylim(bottom=0)
        
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved Figure 2 reproduction to: {output}")


if __name__ == "__main__":
    plot_figure2()