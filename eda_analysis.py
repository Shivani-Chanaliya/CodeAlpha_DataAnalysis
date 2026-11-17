import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


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
    return figure


def main():
    st.set_page_config(page_title="Quotes Exploratory Data Analysis", layout="wide")
    st.title("Quotes Exploratory Data Analysis")
    st.caption("An exploratory look at the collected quotes sample.")

    try:
        data, missing_rows, duplicate_rows, original_rows = load_quotes()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Unable to load the quotes dataset: {error}")
        st.stop()

    data["word_count"] = data["quote"].str.split().str.len()
    data["character_count"] = data["quote"].str.len()

    st.subheader("Data quality")
    metrics = st.columns(4)
    metrics[0].metric("Rows read", f"{original_rows:,}")
    metrics[1].metric("Rows retained", f"{len(data):,}")
    metrics[2].metric("Rows with missing values", f"{missing_rows:,}")
    metrics[3].metric("Duplicate quote-author rows", f"{duplicate_rows:,}")

    chart_column, summary_column = st.columns((2, 1))
    with chart_column:
        st.subheader("Quote and author patterns")
        figure = make_chart(data)
        st.pyplot(figure)
        plt.close(figure)

    with summary_column:
        st.subheader("Quotes by author")
        author_counts = (
            data["author"]
            .value_counts()
            .head(10)
            .rename_axis("Author")
            .reset_index(name="Quotes")
        )
        st.dataframe(author_counts, hide_index=True, width="stretch")
        st.subheader("Numeric summary")
        st.dataframe(
            data[["word_count", "character_count"]].describe().round(2),
            width="stretch",
        )

    if len(data) >= 3:
        rho, p_value = spearman_permutation_test(
            data["word_count"].tolist(),
            data["character_count"].tolist(),
        )
        st.subheader("Word count and character count")
        result_columns = st.columns(2)
        result_columns[0].metric("Spearman rank correlation", f"{rho:.3f}")
        result_columns[1].metric("Two-sided permutation p-value", f"{p_value:.3f}")
        if p_value < 0.05:
            st.write("At the 0.05 level, reject the null hypothesis for this sample.")
        else:
            st.write("At the 0.05 level, do not reject the null hypothesis for this sample.")
        st.caption(
            "This exploratory result describes the scraped sample, not all quotations."
        )
    else:
        st.info("Not enough rows to run the correlation test; at least 3 are needed.")

    with st.expander("View cleaned quote data"):
        st.dataframe(
            data[["quote", "author", "word_count", "character_count"]],
            hide_index=True,
            width="stretch",
        )

    st.caption(f"Chart saved to {CHART_FILE.name}")


if __name__ == "__main__":
    main()
