import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "scraped_quotes.csv"
CHART_FILE = PROJECT_DIR / "top_authors_chart.png"
REQUIRED_COLUMNS = {"quote", "author"}


def load_quotes(data_file=DATA_FILE):
    data = pd.read_csv(data_file)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Dataset has no rows to analyze.")

    for column in REQUIRED_COLUMNS:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace("", pd.NA)

    missing_rows = int(data[list(REQUIRED_COLUMNS)].isna().any(axis=1).sum())
    duplicate_rows = int(data.duplicated(subset=["quote", "author"]).sum())
    clean_data = data.dropna(subset=list(REQUIRED_COLUMNS)).drop_duplicates(
        subset=["quote", "author"]
    )
    if clean_data.empty:
        raise ValueError("No complete quote records remain after data quality checks.")

    return clean_data.copy(), missing_rows, duplicate_rows, len(data)


def correlation(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_sum_squares = sum((value - left_mean) ** 2 for value in left)
    right_sum_squares = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    return numerator / denominator if denominator else 0.0


def spearman_permutation_test(word_counts, character_counts, permutations=999):
    word_ranks = pd.Series(word_counts).rank(method="average").tolist()
    character_ranks = pd.Series(character_counts).rank(method="average").tolist()
    observed_rho = correlation(word_ranks, character_ranks)

    generator = random.Random(42)
    extreme_results = 0
    for _ in range(permutations):
        shuffled_ranks = character_ranks.copy()
        generator.shuffle(shuffled_ranks)
        if abs(correlation(word_ranks, shuffled_ranks)) >= abs(observed_rho):
            extreme_results += 1

    p_value = (extreme_results + 1) / (permutations + 1)
    return observed_rho, p_value


def make_chart(data, output_file=CHART_FILE):
    author_counts = data["author"].value_counts().head(10)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = sns.color_palette("viridis", n_colors=len(author_counts))
    axes[0].bar(author_counts.index, author_counts.values, color=colors)
    axes[0].set_title("Authors with the Most Quotes")
    axes[0].set_xlabel("Author")
    axes[0].set_ylabel("Quotes in sample")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].hist(data["character_count"], bins="auto", color="#6a51a3", edgecolor="white")
    axes[1].set_title("Quote Lengths in the Sample")
    axes[1].set_xlabel("Characters per quote")
    axes[1].set_ylabel("Number of quotes")

    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    data, missing_rows, duplicate_rows, original_rows = load_quotes()
    data["word_count"] = data["quote"].str.split().str.len()
    data["character_count"] = data["quote"].str.len()

    print("Questions")
    print("- Which authors appear most often in this sample?")
    print("- Are word count and character count associated?")
    print("\nData structure")
    print(f"Rows read: {original_rows}")
    print(f"Rows with missing quote or author: {missing_rows}")
    print(f"Duplicate quote-author rows: {duplicate_rows}")
    print(f"Rows retained for analysis: {len(data)}")
    print("\nColumn types")
    print(data[["quote", "author", "word_count", "character_count"]].dtypes)
    print("\nNumeric summary")
    print(data[["word_count", "character_count"]].describe().round(2))
    print("\nQuotes by author")
    print(data["author"].value_counts().head(10).to_string())

    if len(data) >= 3:
        rho, p_value = spearman_permutation_test(
            data["word_count"].tolist(),
            data["character_count"].tolist(),
        )
        print("\nHypothesis test: word count and character count")
        print("H0: the two measures have no rank association.")
        print("H1: the two measures have a rank association.")
        print(f"Spearman rho: {rho:.3f}; two-sided permutation p-value: {p_value:.3f}")
        if p_value < 0.05:
            print("At the 0.05 level, reject H0 for this sample.")
        else:
            print("At the 0.05 level, do not reject H0 for this sample.")
        print("This exploratory result describes the scraped sample, not all quotations.")
    else:
        print("\nNot enough rows to run the correlation hypothesis test (need at least 3).")

    make_chart(data)
    print(f"\nChart saved to {CHART_FILE}")


if __name__ == "__main__":
    main()
