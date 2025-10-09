import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_small_multiples(df_res, output="mse_small_multiples.png"):
    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=df_res, x="df", y="mse", hue="method", palette="Set1"
    )

    plt.xlabel("Degrees of Freedom (t-distribution)")
    plt.ylabel("MSE")
    plt.title("MSE Distribution across Methods and Heavy-Tail Levels")
    plt.legend(title="Method", fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output}")