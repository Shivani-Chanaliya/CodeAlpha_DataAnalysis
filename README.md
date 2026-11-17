# CodeAlpha Exploratory Data Analysis

This project explores the quotes snapshot collected in the Web Scraping task. It inspects the data structure, checks for incomplete and duplicate records, summarizes quote and author patterns, and tests one relationship in the sample.

## Questions

1. Which authors appear most often in the collected sample?
2. Are quote word count and character count associated?

The second question is tested with Spearman's rank correlation and a two-sided permutation test (999 shuffles, fixed seed). The null hypothesis is that the two measures have no rank association. This is exploratory: the quotes are a scraped sample, not a random sample of all writing, and the result should not be generalized beyond it.

## Data checks

The script requires `quote` and `author` columns, reports rows with missing values and duplicate quote-author pairs, and excludes those records before calculating summaries. It adds word and character counts for analysis without changing the source CSV.

## Run

From this folder:

```powershell
python -m pip install pandas matplotlib seaborn
python eda_analysis.py
```

The summary and test result print to the terminal. The script saves `top_authors_chart.png` beside the input CSV.

## View results in Jupyter

Open `eda_analysis.ipynb` in Jupyter Notebook or VS Code and run its code cell. It prints the analysis results first, then displays the charts below them.

## Files

- `eda_analysis.py` — data checks, summaries, hypothesis test, and chart
- `eda_analysis.ipynb` — notebook view with text results followed by the charts
- `scraped_quotes.csv` — input sample
- `top_authors_chart.png` — generated author-count and quote-length charts
