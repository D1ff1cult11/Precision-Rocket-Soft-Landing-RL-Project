import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics() -> None:
    log_file = "logs/training_metrics.csv"
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
    df = pd.read_csv(log_file)
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    df["ma_reward"] = df["avg_reward"].rolling(window=10).mean()
    sns.lineplot(
        data=df,
        x="episode",
        y="avg_reward",
        ax=axes[0],
        alpha=0.3,
        label="Episode Reward",
    )
    sns.lineplot(
        data=df,
        x="episode",
        y="ma_reward",
        ax=axes[0],
        color="blue",
        label="10-Ep Moving Avg",
    )
    axes[0].set_title("Training Reward over Episodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Reward")
    sns.lineplot(data=df, x="episode", y="step", ax=axes[1], color="orange")
    axes[1].set_title("Cumulative Steps over Episodes")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Steps")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_metrics.png", dpi=300)
    print("Plots saved to plots/training_metrics.png")


if __name__ == "__main__":
    plot_metrics()
