#!/usr/bin/env python3
"""
ROC Analysis: Frequency-Based Classification
===========================================

Performs ROC and Precision-Recall analysis using allele frequency
as a proxy for pathogenicity (ultra-rare = pathogenic, common = benign).

Usage:
    python roc_analysis.py --input variant_data.csv --output roc_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import argparse
import sys
from typing import Dict, Any, Tuple


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


def create_roc_analysis(
    df: pd.DataFrame,
    output_path: str,
    rare_threshold: float = 1e-4,
    common_threshold: float = 1e-2,
) -> Dict[str, Any]:
    """
    Create clean ROC and Precision-Recall analysis using frequency-based classification.

    Args:
        df: DataFrame with 'gnomad_af' and 'pathogenicity_score' columns
        output_path: Path where the plot image will be saved
        rare_threshold: Allele frequency threshold for ultra-rare variants
        common_threshold: Allele frequency threshold for common variants

    Returns:
        Dictionary containing ROC analysis results and metrics
    """

    # Create binary labels based on frequency
    # Ultra-rare variants (< rare_threshold) = 1 (pathogenic proxy)
    # Common variants (> common_threshold) = 0 (benign proxy)

    ultra_rare: pd.DataFrame = df[df["gnomad_af"] < rare_threshold].copy()
    common: pd.DataFrame = df[df["gnomad_af"] > common_threshold].copy()

    # Combine datasets
    analysis_df: pd.DataFrame = pd.concat([ultra_rare, common]).copy()
    analysis_df["binary_label"] = (analysis_df["gnomad_af"] < rare_threshold).astype(
        int
    )

    print(
        f"ROC Analysis: {len(ultra_rare):,} ultra-rare vs {len(common):,} common variants"
    )

    # Extract labels and scores
    y_true: np.ndarray = analysis_df["binary_label"].values
    y_scores: np.ndarray = analysis_df["pathogenicity_score"].values

    # Calculate ROC curve
    fpr: np.ndarray
    tpr: np.ndarray
    roc_thresholds: np.ndarray
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc: float = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision: np.ndarray
    recall: np.ndarray
    pr_thresholds: np.ndarray
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc: float = auc(recall, precision)

    # Baseline precision (random classifier)
    baseline_precision: float = y_true.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.style.use("default")

    # Plot 1: ROC curve
    axes[0].plot(
        fpr, tpr, linewidth=2.5, label=f"ROC AUC = {roc_auc:.3f}", color="#2E86C1"
    )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    axes[0].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    axes[0].tick_params(axis="both", which="major", labelsize=11)

    # Plot 2: Precision-recall curve
    axes[1].plot(
        recall,
        precision,
        linewidth=2.5,
        label=f"PR AUC = {pr_auc:.3f}",
        color="#E74C3C",
    )
    axes[1].axhline(
        y=baseline_precision,
        color="k",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
        label=f"Random = {baseline_precision:.3f}",
    )
    axes[1].set_xlabel("Recall", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Precision", fontsize=12, fontweight="bold")
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    axes[1].tick_params(axis="both", which="major", labelsize=11)

    # Plot 3: Score distributions
    axes[2].hist(
        ultra_rare["pathogenicity_score"],
        bins=40,
        alpha=0.7,
        density=True,
        label=f"Ultra-rare (n={len(ultra_rare):,})",
        color="#E74C3C",
        edgecolor="none",
    )
    axes[2].hist(
        common["pathogenicity_score"],
        bins=40,
        alpha=0.7,
        density=True,
        label=f"Common (n={len(common):,})",
        color="#2E86C1",
        edgecolor="none",
    )
    axes[2].set_xlabel("Pathogenicity Score", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Density", fontsize=12, fontweight="bold")
    axes[2].set_title("Score Distributions", fontsize=14, fontweight="bold")
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    axes[2].tick_params(axis="both", which="major", labelsize=11)

    # Clean up all subplots
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"ROC analysis plot saved to: {output_path}")

    # Find optimal threshold using Youden's index
    optimal_idx: int = np.argmax(tpr - fpr)
    optimal_threshold: float = roc_thresholds[optimal_idx]

    # Calculate basic statistics
    ultra_rare_mean: float = ultra_rare["pathogenicity_score"].mean()
    common_mean: float = common["pathogenicity_score"].mean()
    mean_difference: float = ultra_rare_mean - common_mean

    # Print basic results
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Baseline precision: {baseline_precision:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Mean score difference: {mean_difference:.4f}")

    # Return results
    results: Dict[str, Any] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "baseline_precision": baseline_precision,
        "optimal_threshold": optimal_threshold,
        "n_ultra_rare": len(ultra_rare),
        "n_common": len(common),
        "mean_difference": mean_difference,
    }

    return results


def main() -> Dict[str, Any]:
    """
    Main function to parse arguments and run ROC analysis.

    Returns:
        Dictionary containing ROC analysis results and metrics
    """
    parser = argparse.ArgumentParser(
        description="ROC analysis using frequency-based classification"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file with variant data"
    )
    parser.add_argument(
        "--output", "-o", default="roc_analysis.png", help="Output plot filename"
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=1e-4,
        help="Allele frequency threshold for ultra-rare variants (default: 1e-4)",
    )
    parser.add_argument(
        "--common-threshold",
        type=float,
        default=1e-2,
        help="Allele frequency threshold for common variants (default: 1e-2)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show plot in addition to saving"
    )

    args: argparse.Namespace = parser.parse_args()

    # Load data and create analysis
    df: pd.DataFrame = load_data(args.input)
    results: Dict[str, Any] = create_roc_analysis(
        df, args.output, args.rare_threshold, args.common_threshold
    )

    if args.show:
        plt.show()

    return results


if __name__ == "__main__":
    main()
