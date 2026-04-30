#!/usr/bin/env python3
"""
Core Validation: Pathogenicity Score vs. Allele Frequency
=======================================================

Creates a validation plot showing the relationship between variant pathogenicity
scores and gnomAD population frequencies.

Usage:
    python core_validation.py --input variant_data.csv --output core_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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
        required_columns: list[str] = ["variant_id", "pathogenicity_score", "gnomad_af"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Filter out variants with missing or zero frequencies
        df = df[(df["gnomad_af"].notna()) & (df["gnomad_af"] > 0)]

        print(f"Loaded {len(df):,} variants with valid frequency data")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def create_validation_plot(df: pd.DataFrame, output_path: str) -> Tuple[float, float]:
    """
    Create clean scatter plot of pathogenicity scores vs allele frequencies with correlation analysis.

    Args:
        df: DataFrame with 'gnomad_af' and 'pathogenicity_score' columns
        output_path: Path where the plot image will be saved

    Returns:
        Tuple of (correlation_coefficient, p_value) from Spearman correlation test
    """
    plt.figure(figsize=(10, 7))
    plt.style.use("default")

    plt.scatter(
        np.log10(df["gnomad_af"]),
        df["pathogenicity_score"],
        alpha=0.4,
        s=2,
        color="#2E86C1",
        rasterized=True,
        edgecolors="none",
    )

    # Calculate correlation
    result = spearmanr(-np.log10(df["gnomad_af"]), df["pathogenicity_score"])
    correlation: float = float(result[0])
    p_value: float = float(result[1])

    # Add trend line
    z: np.ndarray = np.polyfit(-np.log10(df["gnomad_af"]), df["pathogenicity_score"], 1)
    p: np.poly1d = np.poly1d(z)
    x_trend: np.ndarray = np.linspace(
        np.log10(df["gnomad_af"]).min(), np.log10(df["gnomad_af"]).max(), 100
    )
    plt.plot(
        x_trend, p(-x_trend), color="#E74C3C", linewidth=2.5, alpha=0.8, linestyle="--"
    )

    plt.xlabel("log₁₀(gnomAD Allele Frequency)", fontsize=14, fontweight="bold")
    plt.ylabel("Pathogenicity Score", fontsize=14, fontweight="bold")
    plt.title(
        "Pathogenicity Score vs Population Frequency",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add minimal statistics
    stats_text: str = f"r = {correlation:.3f}, p = {p_value:.1e}\nn = {len(df):,}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9
        ),
        fontsize=12,
        verticalalignment="top",
    )

    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.tick_params(axis="both", which="major", labelsize=12)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Validation plot saved to: {output_path}")

    # Print basic results
    print(f"\nSpearman correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Number of variants: {len(df):,}")

    return correlation, p_value


def main() -> Tuple[float, float]:
    """
    Main function to parse arguments and run core validation analysis.

    Returns:
        Tuple of (correlation_coefficient, p_value) from the validation analysis
    """
    parser = argparse.ArgumentParser(
        description="Create pathogenicity vs frequency validation plot"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file with variant data"
    )
    parser.add_argument(
        "--output", "-o", default="core_validation.png", help="Output plot filename"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show plot in addition to saving"
    )

    args: argparse.Namespace = parser.parse_args()

    # Load data and create plot
    df: pd.DataFrame = load_data(args.input)
    correlation: float
    p_value: float
    correlation, p_value = create_validation_plot(df, args.output)

    if args.show:
        plt.show()

    return correlation, p_value


if __name__ == "__main__":
    main()
