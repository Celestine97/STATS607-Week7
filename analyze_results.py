import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


def theoretical_risk(gamma, r2=5, sigma2=1):
    """
    Compute theoretical risk from Hastie et al. (2022) paper.
    
    For gamma < 1: R = sigma^2 * gamma / (1 - gamma)
    For gamma > 1: R = r^2 * (1 - 1/gamma) + sigma^2 / (gamma - 1)
    
    Parameters:
    -----------
    gamma : array-like
        Aspect ratio p/n
    r2 : float
        Signal strength (beta'beta)
    sigma2 : float
        Noise variance
    
    Returns:
    --------
    risk : array-like
        Theoretical risk values
    """
    gamma = np.asarray(gamma)
    risk = np.zeros_like(gamma, dtype=float)
    
    # Underparameterized regime (gamma < 1)
    mask_under = gamma < 1
    risk[mask_under] = sigma2 * gamma[mask_under] / (1 - gamma[mask_under])
    
    # Overparameterized regime (gamma > 1)
    mask_over = gamma > 1
    risk[mask_over] = r2 * (1 - 1/gamma[mask_over]) + sigma2 / (gamma[mask_over] - 1)
    
    return risk


def load_simulation_results():
    """
    Load all three simulation result files.
    These files are already aggregated with columns: nsim, gamma, MSE_hat, se_MSE
    
    Returns:
    --------
    dict : {nsim: DataFrame}
    """
    regimes = [1, 50, 1000]
    dfs = {}
    
    for nsim in regimes:
        csv_file = f"simulation_results_nsim{nsim}.csv"
        try:
            df = pd.read_csv(csv_file)
            
            print(f"✓ Loaded {csv_file}: {len(df)} gamma values")
            print(f"  Columns: {df.columns.tolist()}")
            
            dfs[nsim] = df
            
        except FileNotFoundError:
            print(f"✗ Error: {csv_file} not found!")
            return None
        except Exception as e:
            print(f"✗ Error loading {csv_file}: {e}")
            return None
    
    return dfs


def plot_figure2(dfs, output="figure2_reproduction.pdf"):
    """
    Reproduce Figure 2 from Hastie et al. (2022).
    
    Creates 3 subplots (one per nsim regime):
    - nsim=1: Faint scatter + LOWESS smoothing for gamma < 1 and gamma > 1
    - nsim=50, 1000: Scatter with error bars (±2*SE)
    - All plots show theoretical risk curve and vertical line at gamma=1
    
    Parameters:
    -----------
    dfs : dict
        Dictionary mapping nsim -> DataFrame with columns [gamma, MSE_hat, se_MSE]
    output : str
        Output filename
    """
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    regimes = [1, 50, 1000]
    
    for idx, nsim in enumerate(regimes):
        ax = axes[idx]
        df = dfs[nsim]
        
        # ============================================================
        # 1. Plot theoretical risk curve
        # ============================================================
        gamma_theory_under = np.linspace(0.1, 0.99, 200)
        gamma_theory_over = np.linspace(1.01, 10, 200)
        gamma_theory = np.concatenate([gamma_theory_under, gamma_theory_over])
        
        risk_theory = theoretical_risk(gamma_theory, r2=5, sigma2=1)
        
        ax.plot(gamma_theory, risk_theory, 'k-', linewidth=2.5, 
                label='Theoretical', zorder=10, alpha=0.8)
        
        # ============================================================
        # 2. Vertical line at gamma=1 (interpolation threshold)
        # ============================================================
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, 
                   alpha=0.5, zorder=1)
        
        # ============================================================
        # 3. Plot estimated risk
        # ============================================================
        if nsim == 1:
            # For nsim=1: faint scatter + LOWESS smoothing
            ax.scatter(df['gamma'], df['MSE_hat'], 
                      alpha=0.15, s=8, color='steelblue', 
                      zorder=2, label='Simulated')
            
            # LOWESS for gamma <= 1
            df_under = df[df['gamma'] <= 1].sort_values('gamma')
            if len(df_under) > 10:
                smoothed_under = lowess(df_under['MSE_hat'], df_under['gamma'], 
                                       frac=0.1, return_sorted=True)
                ax.plot(smoothed_under[:, 0], smoothed_under[:, 1], 
                       'b-', linewidth=2.5, label='LOWESS (γ≤1)', zorder=5)
            
            # LOWESS for gamma >= 1
            df_over = df[df['gamma'] >= 1].sort_values('gamma')
            if len(df_over) > 10:
                smoothed_over = lowess(df_over['MSE_hat'], df_over['gamma'], 
                                      frac=0.1, return_sorted=True)
                ax.plot(smoothed_over[:, 0], smoothed_over[:, 1], 
                       'r-', linewidth=2.5, label='LOWESS (γ≥1)', zorder=5)
        
        else:
            # For nsim>1: scatter with error bars (±2*SE)
            ax.errorbar(df['gamma'], df['MSE_hat'], 
                       yerr=2 * df['se_MSE'],
                       fmt='o', markersize=7, capsize=5, capthick=2,
                       color='steelblue', ecolor='lightsteelblue', 
                       elinewidth=2, alpha=0.8,
                       label='Estimated ±2SE', zorder=3)
        
        # ============================================================
        # 4. Formatting
        # ============================================================
        ax.set_xlabel('γ = p/n', fontsize=13, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=13, fontweight='bold')
        ax.set_title(f'$n_{{sim}}$ = {nsim}', fontsize=14, fontweight='bold')
        
        # Use log scale for x-axis (easier than custom split scale)
        ax.set_xscale('log')
        ax.set_xlim(0.08, 12)
        
        # Set y-axis limits
        if nsim == 1:
            ax.set_ylim(0, 15)
        elif nsim == 50:
            ax.set_ylim(0, 12)
        else:
            ax.set_ylim(0, 10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        # Legend
        ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
        
        # Tick parameters
        ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Figure 2 to: {output}")
    plt.savefig("figure2_reproduction.png", dpi=300, bbox_inches='tight')  # Add this line
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("Reproducing Figure 2 from Hastie et al. (2022)")
    print("="*70)
    
    # Load results
    print("\nLoading simulation results...")
    dfs = load_simulation_results()
    
    if dfs is None:
        print("\n✗ Failed to load data. Run simulations first!")
    else:
        # Print summary
        print("\nData summary:")
        for nsim, df in dfs.items():
            print(f"  nsim={nsim:4d}: {len(df):5d} gamma values, "
                  f"MSE range=[{df['MSE_hat'].min():.3f}, {df['MSE_hat'].max():.3f}]")
        
        # Create figure
        print("\nGenerating Figure 2...")
        plot_figure2(dfs, output="figure2_reproduction.pdf")
        
        print("\n" + "="*70)
        print("✓ Analysis complete!")
        print("="*70)