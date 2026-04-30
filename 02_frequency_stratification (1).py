#!/usr/bin/env python3
"""
Frequency Stratification Box Plot
================================

Creates box plots showing pathogenicity score distributions
across different allele frequency categories.

Usage:
    python frequency_stratification.py --input variant_data.csv --output frequency_stratification.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import argparse
import sys
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate variant data from CSV file.

    Args:
        filepath: Path to CSV file containing variant data

    Returns:
        DataFrame with validated variant data (missing/zero frequencies filtered out)

    Raises:
        ValueError: If required columns are missing
        SystemExit: If file cannot be loaded
    """
    try:
        df: pd.DataFrame = pd.read_csv(filepath)
        required_columns: list[str] = ["variant_ID", "EVE_Index", "log10_af"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Filter out variants with missing or zero frequencies
        df = df[(df["log10_af"].notna()) & (df["log10_af"] > 0)]

        print(f"Loaded {len(df):,} variants with valid frequency data")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def assign_frequency_category(af: float) -> str:
    """
    Assign frequency category label based on allele frequency value.

    Args:
        af: Allele frequency value

    Returns:
        String label for the frequency category
    """
    if af < 1e-5:
        return "Ultra-rare\n(< 10⁻⁵)"
    elif af < 1e-4:
        return "Very rare\n(10⁻⁵ - 10⁻⁴)"
    elif af < 1e-3:
        return "Rare\n(10⁻⁴ - 10⁻³)"
    elif af < 1e-2:
        return "Low freq\n(10⁻³ - 10⁻²)"
    else:
        return "Common\n(≥ 10⁻²)"


def create_frequency_stratification_plot(
    df: pd.DataFrame, output_path: str
) -> Tuple[pd.DataFrame, float]:
    """
    Create box plots showing pathogenicity scores stratified by frequency categories.

    Args:
        df: DataFrame with 'gnomad_af' and 'pathogenicity_score' columns
        output_path: Path where the plot image will be saved

    Returns:
        Tuple of (summary_statistics_df, kruskal_p_value)
    """

    # Assign frequency categories
    df["freq_category"] = df["log10_af"].apply(assign_frequency_category)

    # Define category order
    category_order: list[str] = [
        "Ultra-rare\n(< 10⁻⁵)",
        "Very rare\n(10⁻⁵ - 10⁻⁴)",
        "Rare\n(10⁻⁴ - 10⁻³)",
        "Low freq\n(10⁻³ - 10⁻²)",
        "Common\n(≥ 10⁻²)",
    ]

    # Filter to only include categories present in data
    available_categories: list[str] = [
        cat for cat in category_order if cat in df["freq_category"].unique()
    ]

    plt.figure(figsize=(12, 8))
    plt.style.use("default")

    # Create box plot
    ax = sns.boxplot(
        data=df,
        x="freq_category",
        y="EVE_Index",
        order=available_categories,
        palette="viridis",
        linewidth=1.5,
    )

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.ylabel("EVE_Index", fontsize=14, fontweight="bold")
    plt.xlabel("Allele Frequency Category", fontsize=14, fontweight="bold")
    plt.title(
        "Pathogenicity Scores by Frequency Category",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add sample sizes above each box
    category_counts: pd.Series = df["freq_category"].value_counts()

    for i, category in enumerate(available_categories):
        if category in category_counts:
            count: int = category_counts[category]
            plt.text(
                i,
                plt.ylim()[1] * 0.95,
                f"n = {count:,}",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
            )

    plt.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)
    plt.tick_params(axis="both", which="major", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Frequency stratification plot saved to: {output_path}")

    # Calculate basic statistics
    summary_stats: pd.DataFrame = (
        df.groupby("freq_category")["Eve_Index"]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )
    summary_stats.columns = ["Count", "Mean", "Median", "Std"]
    summary_stats = summary_stats.reindex(available_categories)

    # Statistical test
    category_data: list[np.ndarray] = [
        df[df["freq_category"] == cat]["Eve_Index"].values
        for cat in available_categories
    ]
    result = kruskal(*category_data)
    h_stat: float = float(result[0])
    p_value: float = float(result[1])

    # Print basic results
    print(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Number of categories: {len(available_categories)}")

    return summary_stats, p_value


def main() -> Tuple[pd.DataFrame, float]:
    """
    Main function to parse arguments and run frequency stratification analysis.

    Returns:
        Tuple of (summary_statistics_df, kruskal_p_value)
    """
    parser = argparse.ArgumentParser(
        description="Create pathogenicity scores by frequency category box plot"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file with variant data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="frequency_stratification.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show plot in addition to saving"
    )

    args: argparse.Namespace = parser.parse_args()

    # Load data and create plot
    df: pd.DataFrame = load_data(args.input)
    summary_stats: pd.DataFrame
    p_value: float
    summary_stats, p_value = create_frequency_stratification_plot(df, args.output)

    if args.show:
        plt.show()

    return summary_stats, p_value


if __name__ == "__main__":
    main()
