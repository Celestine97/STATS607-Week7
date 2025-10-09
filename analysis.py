import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
def plot_mse_vs_df(df_summary, output="mse_vs_df.png"):
    plt.figure(figsize=(6, 4))

    sns.lineplot(
        data=df_summary,
        x="df", y="mse", hue="method",
        marker="o", linewidth=2.5, markersize=8, palette="Set1"
    )

    plt.xlabel("Degrees of Freedom (t-distribution)", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title("MSE vs Degrees of Freedom for Robust Regression Methods", fontsize=13)
    plt.legend(title="Method", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output}")
'''
def plot_small_multiples(df_res, output="mse_small_multiples.png"):
    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=df_res, x="df", y="mse", hue="method", palette="Set1"
    )

    plt.xlabel("Degrees of Freedom (t-distribution)")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.title("MSE Distribution across Methods and Heavy-Tail Levels")
    plt.legend(title="Method", fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output}")


def plot_mse_vs_df(df_summary, output="mse_vs_df.png"):
    """
    Plot MSE vs Degrees of Freedom (df) for each method,
    faceted by Signal-to-Noise Ratio (SNR).
    """

    g = sns.FacetGrid(
        df_summary,
        col="SNR",
        hue="method",
        col_wrap=3,    
        height=3.5,
        sharey=False,
        palette="Set1"
    )

    g.map_dataframe(
        sns.lineplot,
        x="df",
        y="mse",
        marker="o",
        linewidth=2.0,
        markersize=6
    )
    for ax in g.axes.flat:
        ax.set_yscale("log")
    g.add_legend(title="Method")
    g.set_axis_labels("Degrees of Freedom (t-distribution)", "Mean Squared Error")
    g.set_titles("SNR = {col_name}")
    plt.tight_layout()

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure faceted by SNR to {output}")