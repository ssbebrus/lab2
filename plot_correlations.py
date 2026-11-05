from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    root = Path(__file__).resolve().parent
    output = root / "plots"
    output.mkdir(exist_ok=True)
    frame = pd.read_csv(root / "brainsize.txt", sep="\t", na_values="NA")
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    for row, gender in enumerate(["Male", "Female"]):
        group = frame.loc[frame["Gender"] == gender]
        for column, feature in enumerate(features):
            pair = group[[feature, "MRI_Count"]].dropna()
            correlation = pair[feature].corr(pair["MRI_Count"])
            sns.scatterplot(data=pair, x=feature, y="MRI_Count", ax=axes[row, column])
            axes[row, column].set_title(f"{gender}: r={correlation:.3f}, n={len(pair)}")
    fig.savefig(output / "scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, gender in zip(axes, ["Male", "Female"]):
        matrix = frame.loc[frame["Gender"] == gender, features + ["MRI_Count"]].corr()
        sns.heatmap(matrix, annot=True, fmt=".2f", vmin=-1, vmax=1,
                    cmap="coolwarm", square=True, ax=axis)
        axis.set_title(gender)
    fig.savefig(output / "heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
