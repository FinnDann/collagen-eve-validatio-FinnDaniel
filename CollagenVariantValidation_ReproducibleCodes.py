#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import all necessary libraries
from typing import Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde, kruskal
import seaborn as sns
import math
import scikit_posthocs as sp
from itertools import combinations
import itertools
from sklearn.metrics import accuracy_score, roc_curve, auc

# ============================================================================
# Data Loading and Curation Functions
# ============================================================================
def load_data(collagen_path: str, gnomad_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load collagen and gnomAD datasets.
    
    Args:
        collagen_path: Path to collagen AI predictions CSV
        gnomad_path: Path to gnomAD data CSV
    
    Returns:
        Tuple of (collagen_df, gnomad_df)
    """
    collagen_df = pd.read_csv(collagen_path)
    gnomad_df = pd.read_csv(gnomad_path)
    
    print(f"AI predictions shape: {collagen_df.shape}")
    print(f"gnomAD data shape: {gnomad_df.shape}")
    
    return collagen_df, gnomad_df

def merge_collagen_with_gnomad_variants(
    collagen_df: pd.DataFrame,
    gnomad_df: pd.DataFrame,
    merge_keys: list = ["Chromosome", "Position", "Reference", "Alternate"],
    how: str = "left"
) -> pd.DataFrame:
    """
    Merge collagen predictions with gnomAD variant data.
    """
    # Ensure merge keys are of string type in both DataFrames
    for key in merge_keys:
        if key in gnomad_df.columns and key in collagen_df.columns:
            gnomad_df[key] = gnomad_df[key].astype(str)
            collagen_df[key] = collagen_df[key].astype(str)
        else: 
            print(f" Merge key '{key}' missing in one of the dataframes.")
            return pd.DataFrame()

    # merge
    merged = pd.merge(
        collagen_df,
        gnomad_df,
        on=merge_keys,
        how=how
    )

    # Compute log10(gnomad AF) if present
    if "Allele Frequency" in merged.columns:
        merged = merged[(merged["Allele Frequency"].notna()) & (merged["Allele Frequency"] > 0)]
        merged["log10_af"] = -np.log10(merged["Allele Frequency"] + 1e-6)
    
    # Rename gene column if both have it
    if "gene_x" in merged.columns:
        merged.rename(columns={"gene_x": "gene"}, inplace=True)

    print(f" Merge complete: {merged.shape[0]} rows matched.")
    return merged

def filter_uncertain_clinsig(df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Filter out variants with uncertain clinical significance.
    
    Args:
        df: Merged dataframe with 'clinsig' column
        output_path: Optional path to save filtered dataframe
    
    Returns:
        Filtered dataframe with uncertain variants removed
    """
    # Count before filtering
    original_count = len(df)
    original_genes = df['gene'].nunique() if 'gene' in df.columns else 0
    
    # Filter out rows with "uncertain" in clinsig
    filtered_df = df[~df["clinsig"].str.lower().str.contains("uncertain", na=False)].copy()
    
    # Count after filtering
    filtered_count = len(filtered_df)
    filtered_genes = filtered_df['gene'].nunique() if 'gene' in filtered_df.columns else 0
    
    # Print summary
    print(f"\n{'='*50}")
    print("Filtering uncertain clinical significance variants:")
    print(f"{'='*50}")
    print(f"Rows before filtering: {original_count:,}")
    print(f"Rows after filtering: {filtered_count:,}")
    print(f"Rows removed: {original_count - filtered_count:,} ({((original_count - filtered_count)/original_count)*100:.1f}%)")
    print(f"Unique genes before: {original_genes}")
    print(f"Unique genes after: {filtered_genes}")
    print(f"{'='*50}\n")
    
    # Save
    if output_path:
        filtered_df.to_csv(output_path, index=False)
        print(f"Filtered dataframe saved to: {output_path}")
    
    return filtered_df


# ============================================================================
# Analyzing and Visualization Functions
# ============================================================================

def plot_clinsig_distribution(df: pd.DataFrame, gene: Union[str, None] = None) -> None:
    """
    Plots the frequency of clinsig categories (y-axis) across all genes or a specific gene.
    """
    raw_to_pretty = {
        "benign": "Benign",
        "likely_benign": "Likely benign",
        "uncertain": "Uncertain significance",
        "conflicting": "Conflicting significance",
        "path": "Pathogenic",
        "likely_path": "Likely pathogenic"
    }

    # Apply gene filtering if specified
    if gene:
        df = df[df["gene"].str.upper() == gene.upper()]
        title_prefix = f"{gene} - "
    else:
        title_prefix = ""

    # Map to pretty labels
    df = df[df["clinsig"].isin(raw_to_pretty.keys())].copy()
    df["clinsig_pretty"] = df["clinsig"].map(raw_to_pretty)

    # Count frequency
    clinsig_counts = df["clinsig_pretty"].value_counts().reindex([
        "Benign", "Likely benign", "Likely pathogenic", "Pathogenic",
        "Uncertain significance", "Conflicting significance"
    ])
    
    # Calculate percentages
    total_count = clinsig_counts.sum()
    percentages = (clinsig_counts / total_count * 100).round(1)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Summary for: {title_prefix if gene else 'All genes'}")
    print(f"{'='*50}")
    print(f"Total variants: {total_count}\n")
    print("Breakdown by clinical significance:")
    print("-" * 40)
    for category, count, pct in zip(clinsig_counts.index, clinsig_counts.values, percentages.values):
        if not pd.isna(count):
            print(f"{category:25} n={int(count):4} ({pct:5.1f}%)")
    print(f"{'='*50}\n")

    # Plot (only percentages on bars now)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=clinsig_counts.index,
        y=clinsig_counts.values,
        palette="muted"
    )
    
    # Add percentage labels above each bar
    max_count = clinsig_counts.max()
    for i, (count, percentage) in enumerate(zip(clinsig_counts.values, percentages.values)):
        if not pd.isna(count):
            ax.text(i, count + (max_count * 0.02), 
                   f'{percentage}%', 
                   ha='center', 
                   va='bottom',
                   fontsize=9,
                   fontweight='bold')
    
    plt.ylim(0, max_count * 1.2)
    plt.ylabel("Variant Count", fontsize=12)
    plt.xlabel("Clinical Significance", fontsize=12)
    plt.title(f"{title_prefix}ClinVar Classification Frequency", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=25)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_violin_eve_by_clinsig(df: pd.DataFrame, score_col: str = "EVE_index", gene: Union[str, None] = None) -> None:
    """
    Create a violin plot showing the distribution of a score across clinsig categories.
    """
    raw_to_pretty = {
        "benign": "Benign",
        "likely_benign": "Likely benign",
        "uncertain": "Uncertain significance",
        "conflicting": "Conflicting significance",
        "path": "Pathogenic",
        "likely_path": "Likely pathogenic"
    }

    ordered_labels = [
        "Benign",
        "Likely benign",
        "Likely pathogenic",
        "Pathogenic",
        "Uncertain significance",
        "Conflicting significance"
    ]
        
    custom_palette = {
        "Benign": "forestgreen",
        "Likely benign": "limegreen",
        "Likely pathogenic": "orangered",
        "Pathogenic": "red",
        "Uncertain significance": "silver",
        "Conflicting significance": "gray"
    }
    
    # Apply gene filtering if specified
    if gene:
        df = df[df["gene"].str.upper() == gene.upper()]
        title_suffix = f" for {gene}"
    else:
        title_suffix = ""
        
    # Map raw clinsig to pretty labels
    df = df[df["clinsig"].isin(raw_to_pretty.keys())].copy()
    df["clinsig_pretty"] = df["clinsig"].map(raw_to_pretty)
    df["clinsig_pretty"] = pd.Categorical(df["clinsig_pretty"], categories=ordered_labels, ordered=True)

    # Filter for non-missing EVE_index
    df_filtered = df[df[score_col].notna()]
    
    # Plot 
    plt.figure(figsize=(6, 6))
    sns.violinplot(
        y="clinsig_pretty",
        x=score_col,
        data=df_filtered,
        order=ordered_labels,
        palette=custom_palette,
        orient="h"
    )
    
    plt.ylabel("ClinVar Clinical Significance", fontsize=13, fontweight="bold")
    plt.xlabel(f"{score_col}", fontsize=13, fontweight="bold")
    plt.title(f"{score_col} Distribution by ClinVar Annotations{title_suffix}", fontsize=15, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_hist_distribution_plots_total(hist_data_total: pd.DataFrame, column_name: str) -> None:
    """
    Create histogram distribution plot for a column (such as the EVE index) across the merged data.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(hist_data_total[column_name].dropna(), bins=20, kde=False, color="cornflowerblue")
    plt.title(f"Distribution - {column_name}")
    plt.xlabel("$-\log_{10}(\mathrm{AF})$")
    plt.ylabel("Col Gene Variant Count")
    plt.grid(True)
    plt.show()

def create_hist_distribution_plots_individual(hist_data_indiv: pd.DataFrame, column_name: str, cols: int=3, output_filename: str = None): 
    """
    Create individual histogram distribution plots (such as for EVE index and gnomad AF) per gene with merged data.
    """
    genes = hist_data_indiv['gene'].unique()
    num_genes = len(genes)
    rows = math.ceil(num_genes/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, gene_id in enumerate(genes):
        gene_data = hist_data_indiv[hist_data_indiv['gene'] == gene_id]
        
        ax = axes[i]
        sns.histplot(gene_data[column_name].dropna(),
                     bins=20, ax=ax,
                     kde=True
                    )      
        ax.set_title(gene_id, fontsize=9)
        ax.set_xlabel('EVE Index')
        ax.set_ylabel('Variant Count') 
        ax.grid(True)

    # Hide unused subplots
    for i in range(num_genes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{column_name} Distributions Across {num_genes} Gene Variants", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved individual histogram to: {output_filename}")
    
    plt.show()

def plot_eve_histogram_with_af_overlay(
    collagen_df: pd.DataFrame,
    gnomad_df: pd.DataFrame,
    score_col: str = "EVE_index",
    clinsig_col: str = "clinsig",
    af_col: str = "Allele Frequency"
) -> None:
    """
    Plot bimodal EVE index histogram with common gnomAD AF overlay.
    """
    merge_keys = ["Position", "Reference", "Alternate"]

    # Normalize merge keys to string and uppercase alleles to improve matching
    for key in merge_keys:
        collagen_df[key] = collagen_df[key].astype(str)
        gnomad_df[key] = gnomad_df[key].astype(str)
    for allele_col in ["Reference", "Alternate", "Position"]:
        collagen_df[allele_col] = collagen_df[allele_col].str.upper()
        gnomad_df[allele_col] = gnomad_df[allele_col].str.upper()

    # Merge only to add AF, keep all collagen_df rows (left join)
    merged = pd.merge(
        collagen_df,
        gnomad_df[merge_keys + [af_col]],
        on=merge_keys,
        how="left"
    )

    # Filter AI predictions for pathogenic and benign only
    df_filtered = merged[
        merged[clinsig_col].isin(["path", "benign"]) &
        merged[score_col].notna()
    ].copy()

    # Separate by clinsig
    path_df = df_filtered[df_filtered[clinsig_col] == "path"]
    benign_df = df_filtered[df_filtered[clinsig_col] == "benign"]

    # Subset variants with AF > 0.01 for KDE overlay
    high_af_df = merged[
        (merged[af_col] > 0.01) & 
        merged[score_col].notna()
    ]

    # Prepare x-axis for KDE
    x_vals = np.linspace(df_filtered[score_col].min(), df_filtered[score_col].max(), 1000)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot pathogenic histogram
    sns.histplot(
        path_df[score_col], bins=40, color="orangered", alpha=0.6,
        label=f"Pathogenic (n={len(path_df)})", stat="count", ax=ax
    )

    # Plot benign histogram
    sns.histplot(
        benign_df[score_col], bins=40, color="cornflowerblue", alpha=0.6,
        label=f"Benign (n={len(benign_df)})", stat="count", ax=ax
    )

    # KDE overlay for variants with AF > 0.01
    if len(high_af_df) > 1:
        kde = gaussian_kde(high_af_df[score_col])
        y_vals = kde(x_vals) * len(high_af_df)
        ax.plot(x_vals, y_vals, color="purple", lw=2,
                label=f"gnomAD AF > 0.01 (n={len(high_af_df)})")

    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("EVE Index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Variant Count", fontsize=14, fontweight="bold")
    ax.set_title("EVE Index Distribution with gnomAD AF Overlay", fontsize=15, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_validation_plot(
    df: pd.DataFrame,
    af_col: str = "log10_af",
    score_col: str = "pathogenicity_score",
    output_path: str = "validation_plot.png"
) -> Tuple[float, float]:
    """
    Create ultimate validation plot comparing score (such as EVE index) vs allele frequency.
    """
    df_filtered = df[(df[af_col].notna()) & (df[score_col].notna())]

    plt.figure(figsize=(10, 7))
    plt.style.use("default")

    # Bin the data
    df_filtered["bin"] = pd.cut(df_filtered[af_col], bins=15)
    grouped = df_filtered.groupby("bin")
    bin_centers = grouped[af_col].mean().to_numpy()
    bin_means = grouped[score_col].mean().to_numpy()
    bin_stds = grouped[score_col].std().to_numpy()
    
    plt.plot(bin_centers, bin_means, color="darkviolet", lw=2, label="Mean")

    # Plot standard deviation shaded area
    plt.fill_between(
        bin_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        color="steelblue",
        alpha=0.3,
        label="±1 SD"
    )

    plt.legend()

    # Spearman correlation
    result = spearmanr(df_filtered["log10_af"], df_filtered[score_col])
    correlation = float(result[0])
    p_value = float(result[1])

    plt.xlabel(r'$-\log_{10}(\mathrm{AF})$', fontsize=14, fontweight="bold")
    plt.ylabel("Eve Index", fontsize=14, fontweight="bold")
    plt.title(f"{score_col} vs Allele Frequency", fontsize=16, fontweight="bold", pad=20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    stats_text = f"Spearman r = {correlation:.3f}, p = {p_value:.1e}\nn = {len(df_filtered):,}"
    plt.text(
        0.02, 0.98, stats_text, transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        fontsize=12, verticalalignment="top",
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
    plt.show()

    print(f"Validation plot saved to: {output_path}")
    print(f"Spearman correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Number of variants plotted: {len(df_filtered):,}")

    return correlation, p_value

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

def cliffs_delta_np(x, y):
    """
    Compute Cliff's delta and interpret its magnitude.
    
    Args:
        x, y: arrays of values for two groups
        
    Returns:
        delta: Cliff's delta value (ranges from -1 to 1)
        magnitude: string describing effect size (negligible/small/medium/large)
    """
    nx = len(x)
    ny = len(y)
    
    # Convert to arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Count pairwise comparisons
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    
    # Calculate Cliff's delta
    delta = (more - less) / (nx * ny)
    
    # Interpret magnitude (using common thresholds)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return delta, magnitude

def create_frequency_stratification_plot(df: pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, float]:
    """
    Create horizontal box plots of the selected pathogenicity score across allele frequency categories,
    with space for Cliff's delta annotations.

    Args:
        df: DataFrame with 'Allele Frequency' and the selected score column
        column_name: The name of the column to be plotted on the x-axis

    Returns:
        Tuple of (summary_statistics_df, kruskal_p_value)
    """
    df["freq_category"] = df["Allele Frequency"].apply(assign_frequency_category)

    category_order = [
        "Ultra-rare\n(< 10⁻⁵)",
        "Very rare\n(10⁻⁵ - 10⁻⁴)",
        "Rare\n(10⁻⁴ - 10⁻³)",
        "Low freq\n(10⁻³ - 10⁻²)",
        "Common\n(≥ 10⁻²)",
    ]
    available_categories = [cat for cat in category_order if cat in df["freq_category"].unique()]

    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    sns.set_style("whitegrid")  # Or "darkgrid", "white", "dark", "ticks"

    palette = sns.color_palette("viridis", n_colors=len(available_categories))

    sns.boxplot(
        data=df,
        y="freq_category",
        x=column_name,
        order=available_categories,
        palette=palette,
        linewidth=1.5,
        width=0.6,
        fliersize=4,
        orient="h",
        saturation=0.8,
        boxprops=dict(alpha=0.9),
        ax=ax
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


    ax.set_yticklabels(available_categories, fontsize=12, fontweight='semibold')
    ax.set_xlabel(column_name, fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Allele Frequency Category", fontsize=14, fontweight="bold", labelpad=10)


    category_counts = df["freq_category"].value_counts()
    x_max = df[column_name].max()
    x_min = df[column_name].min()


    # Kruskal-Wallis test
    category_data = [
        df[df["freq_category"] == cat][column_name].dropna().values
        for cat in available_categories
        if len(df[df["freq_category"] == cat][column_name].dropna()) > 0
    ]

    if len(category_data) >= 2:
        h_stat, p_value = kruskal(*category_data)
        n_total = sum(len(group) for group in category_data)
        k = len(category_data)
        epsilon_sq = (h_stat - k + 1) / (n_total - k) if n_total > k else np.nan
        kruskal_text = f"Kruskal-Wallis\nH = {h_stat:.2f}\np = {p_value:.2e}\nε² = {epsilon_sq:.3f}"
        print(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}\nP-value: {p_value:.2e}\nEpsilon squared: {epsilon_sq:.4f}")
    else:
        h_stat, p_value, epsilon_sq = np.nan, np.nan, np.nan
        kruskal_text = "Kruskal-Wallis\ntest not valid"
        print("\n Not enough non-empty groups to run Kruskal-Wallis test.")

    ax.text(
        0.01, 1.06, kruskal_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="normal",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1, alpha=0.9)
    )

    # Improved Cliff's delta annotations with arrows
    def add_delta_arrow(y1, y2, label, side="left", color="black"):
        # Ensure offset is never too small
        arrow_length = 0.02 * (x_max - x_min)
        if abs(x_min) < 1e-3:
            x_min_pos = -arrow_length * 3
        else:
            x_min_pos = x_min * 0.95

        x_pos = x_min_pos if side == "left" else x_max * 0.95

        arrowprops = dict(
            arrowstyle='<->',
            color=color,
            lw=1.5,
            shrinkA=0,
            shrinkB=0,
            linestyle='-'
        )

        ax.annotate(
            "", 
            xy=(x_pos, y2),
            xytext=(x_pos, y1),
            arrowprops=arrowprops,
            annotation_clip=False
        )

        text_x = x_pos - (arrow_length if side == "left" else -arrow_length)
        ax.text(
            text_x, 
            (y1 + y2)/2, 
            f"Δ = {label}",
            ha="right" if side == "left" else "left",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
        )

    cat_to_idx = {cat: idx for idx, cat in enumerate(available_categories)}

    # Dynamically compute Cliff's delta between ultra-rare and common
    ultra_rare_data = df[df["freq_category"] == "Ultra-rare\n(< 10⁻⁵)"][column_name].dropna()
    common_data = df[df["freq_category"] == "Common\n(≥ 10⁻²)"][column_name].dropna()
    
    if len(ultra_rare_data) > 0 and len(common_data) > 0:
        delta, magnitude = cliffs_delta_np(ultra_rare_data, common_data)
        cliffs_label = f"{delta:.3f} ({magnitude})"
    else:
        cliffs_label = "N/A"
        print("Warning: Could not compute Cliff's delta - insufficient data in extreme categories")
    
    cliffs_annotations = [("Ultra-rare\n(< 10⁻⁵)", "Common\n(≥ 10⁻²)", cliffs_label, "right", "#FF4500")]

    for group1, group2, label, side, color in cliffs_annotations:
        if group1 in cat_to_idx and group2 in cat_to_idx:
            y1 = cat_to_idx[group1]
            y2 = cat_to_idx[group2]
            add_delta_arrow(y1, y2, label, side=side, color=color)

    plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.grid(True, alpha=0.2, axis="x", linestyle="-", linewidth=0.5)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    ax.set_title(
        f"{column_name} Distribution Across Allele Frequency Bins\nn = {n_total} | H = {h_stat:.2f}, p = {p_value:.3e}, ε² = {epsilon_sq:.3f}",
        fontsize=12, fontweight="bold")

    print("\nSample size per frequency category:")
    for category in available_categories:
        n = len(df[df["freq_category"] == category])
        print(f"{category}: n = {n}")


    summary_stats = (
        df.groupby("freq_category")[column_name]
        .agg(["count", "median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .rename(columns={"<lambda_0>": "Q1", "<lambda_1>": "Q3"})
        .round(4)
        .reindex(available_categories)
    )
    summary_stats["IQR"] = summary_stats["Q3"] - summary_stats["Q1"]

    print("\nSummary statistics (Median, Q1, Q3, IQR):")
    print(summary_stats[["median", "Q1", "Q3", "IQR"]])

    return summary_stats, p_value

def find_optimal_threshold(df):
    """
    Find the optimal EVE_index threshold that best separates pathogenic from benign.
    Uses ROC curve analysis to find the threshold that maximizes sensitivity + specificity.
    """
    # Convert labels to binary (1 for Pathogenic, 0 for Benign)
    y_true = (df['CLNSIG'] == 'Pathogenic').astype(int)
    eve_scores = df['EVE_index'].values
    
    # Additional check for NaN values
    if np.any(np.isnan(eve_scores)):
        raise ValueError("EVE_index contains NaN values. Please clean the data first.")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, eve_scores)
    
    # Find optimal threshold (maximizing Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Alternative method: find threshold that maximizes accuracy
    accuracies = []
    for threshold in thresholds:
        y_pred = (eve_scores >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)
    
    max_acc_idx = np.argmax(accuracies)
    threshold_max_acc = thresholds[max_acc_idx]
    
    print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"Optimal threshold (Max Accuracy): {threshold_max_acc:.4f}")
    
    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx]

def plot_threshold_analysis(df, gene_name="COL1A1", output_dir="."):
    """
    Plot threshold analysis for a specific gene comparing pathogenic vs benign.
    """
    # Filter the dataframe
    conditions = (df['gene'] == gene_name) & (df['CLNSIG'].isin(['Pathogenic', 'Benign']))
    filtered_df = df[conditions].copy()
    
    # Remove rows with missing EVE_index values
    print(f"Initial filtered data: {len(filtered_df)} mutations")
    filtered_df = filtered_df.dropna(subset=['EVE_index'])
    print(f"After removing missing EVE_index values: {len(filtered_df)} mutations")
    
    # Separate pathogenic and benign data
    pathogenic_df = filtered_df[filtered_df['CLNSIG'] == 'Pathogenic']
    benign_df = filtered_df[filtered_df['CLNSIG'] == 'Benign']
    
    print(f"\nPathogenic mutations: {len(pathogenic_df)}")
    print(f"Benign mutations: {len(benign_df)}")
    print(f"\nEVE_index statistics:")
    print(f"Pathogenic - min: {pathogenic_df['EVE_index'].min():.4f}, max: {pathogenic_df['EVE_index'].max():.4f}, mean: {pathogenic_df['EVE_index'].mean():.4f}")
    print(f"Benign - min: {benign_df['EVE_index'].min():.4f}, max: {benign_df['EVE_index'].max():.4f}, mean: {benign_df['EVE_index'].mean():.4f}")
    
    # Check if we have data for both classes
    if len(pathogenic_df) == 0 or len(benign_df) == 0:
        raise ValueError("Need at least one example of each class (Pathogenic and Benign) to create the plot.")
    
    # Calculate the optimal threshold
    optimal_threshold, tpr_opt, fpr_opt = find_optimal_threshold(filtered_df)
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Create histogram for pathogenic mutations (red)
    n_path, bins_path, patches_path = plt.hist(pathogenic_df['EVE_index'], 
                                               bins=30, alpha=0.5, color='red', 
                                               label='Pathogenic', edgecolor='darkred')
    
    # Create histogram for benign mutations (green)
    n_benign, bins_benign, patches_benign = plt.hist(benign_df['EVE_index'], 
                                                     bins=30, alpha=0.5, color='green', 
                                                     label='Benign', edgecolor='darkgreen')
    
    # Get current axes for histogram
    ax1 = plt.gca()
    
    # Create a twin y-axis for the scatter plot
    ax2 = ax1.twinx()
    
    # Calculate y-positions for scatter plot (normalized to histogram height)
    max_freq = max(max(n_path), max(n_benign))
    scatter_y_pathogenic = [max_freq * 0.8] * len(pathogenic_df)
    scatter_y_benign = [max_freq * 0.3] * len(benign_df)
    
    # Plot scatter points for pathogenic (red)
    ax2.scatter(pathogenic_df['EVE_index'], scatter_y_pathogenic, 
                c='red', s=50, edgecolors='darkred', linewidth=0.5, 
                alpha=0.8, zorder=5)
    
    # Plot scatter points for benign (green)
    ax2.scatter(benign_df['EVE_index'], scatter_y_benign, 
                c='green', s=50, edgecolors='darkgreen', linewidth=0.5, 
                alpha=0.8, zorder=5)
    
    # Add vertical line at optimal threshold
    ax1.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(optimal_threshold, max_freq * 0.95, f'{optimal_threshold:.2f}', 
             ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax1.set_xlabel('EVE_index', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax2.set_ylabel('CLNSIG', fontsize=14)
    plt.title(f'CLNSIG vs. EVE_index for {gene_name} and Histogram', fontsize=16)
    
    # Configure second y-axis for scatter plot labels
    ax2.set_ylim(0, max_freq)
    ax2.set_yticks([scatter_y_benign[0], scatter_y_pathogenic[0]])
    ax2.set_yticklabels(['Benign', 'Pathogenic'], fontsize=12)
    
    # Add legend for histograms
    ax1.legend(loc='upper left', fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure
    file_name = f"{output_dir}/{gene_name}_EVE_index_SIG-db_with_histogram.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    # Add threshold and classification information to the dataframe
    filtered_df['optimal_threshold'] = optimal_threshold
    filtered_df['predicted_class'] = filtered_df['EVE_index'].apply(
        lambda x: 'Pathogenic' if x >= optimal_threshold else 'Benign'
    )
    filtered_df['correctly_classified'] = filtered_df['CLNSIG'] == filtered_df['predicted_class']
    
    # Save the filtered data to CSV
    csv_filename = f"{output_dir}/{gene_name}_filtered_pathogenic_benign_data.csv"
    filtered_df.to_csv(csv_filename, index=False)
    print(f"\nFiltered data saved to: {csv_filename}")
    print(f"CSV includes {len(filtered_df)} mutations with threshold and classification results")
    
    # Display the plot
    plt.show()
    
    # Print statistics and classification performance
    print(f"\nStatistics for {gene_name}:")
    print(f"Total Pathogenic mutations: {len(pathogenic_df)}")
    print(f"Total Benign mutations: {len(benign_df)}")
    print(f"\nOptimal EVE_index threshold: {optimal_threshold:.4f}")
    print(f"Sensitivity (True Positive Rate) at threshold: {tpr_opt:.3f}")
    print(f"Specificity (1 - False Positive Rate) at threshold: {1-fpr_opt:.3f}")
    
    # Calculate classification results at optimal threshold
    pathogenic_above = len(pathogenic_df[pathogenic_df['EVE_index'] >= optimal_threshold])
    pathogenic_below = len(pathogenic_df[pathogenic_df['EVE_index'] < optimal_threshold])
    benign_above = len(benign_df[benign_df['EVE_index'] >= optimal_threshold])
    benign_below = len(benign_df[benign_df['EVE_index'] < optimal_threshold])
    
    print(f"\nClassification results at threshold {optimal_threshold:.4f}:")
    print(f"Pathogenic mutations correctly classified (≥ threshold): {pathogenic_above} ({pathogenic_above/len(pathogenic_df)*100:.1f}%)")
    print(f"Pathogenic mutations misclassified (< threshold): {pathogenic_below} ({pathogenic_below/len(pathogenic_df)*100:.1f}%)")
    print(f"Benign mutations correctly classified (< threshold): {benign_below} ({benign_below/len(benign_df)*100:.1f}%)")
    print(f"Benign mutations misclassified (≥ threshold): {benign_above} ({benign_above/len(benign_df)*100:.1f}%)")
    print(f"\nOverall accuracy: {(pathogenic_above + benign_below) / len(filtered_df) * 100:.1f}%")
    
    # Create and save a summary dataframe
    summary_data = {
        'gene': [gene_name],
        'optimal_threshold': [optimal_threshold],
        'sensitivity_tpr': [tpr_opt],
        'specificity': [1-fpr_opt],
        'total_pathogenic': [len(pathogenic_df)],
        'total_benign': [len(benign_df)],
        'pathogenic_correctly_classified': [pathogenic_above],
        'pathogenic_misclassified': [pathogenic_below],
        'benign_correctly_classified': [benign_below],
        'benign_misclassified': [benign_above],
        'overall_accuracy': [(pathogenic_above + benign_below) / len(filtered_df)],
        'pathogenic_accuracy': [pathogenic_above/len(pathogenic_df)],
        'benign_accuracy': [benign_below/len(benign_df)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{output_dir}/{gene_name}_EVE_analysis_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nAnalysis summary saved to: {summary_filename}")
    
    return filtered_df, summary_df

# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    # Configuration
    COLLAGEN_PATH = "/home/finn/collagen_colrenamed.csv"
    GNOMAD_PATH = "/home/finn/collagen_colrenamed_deboraversionhm.csv"
    OUTPUT_DIR = "analysis_outputs"

    print("=" * 60)
    print("STARTING COLLAGEN ANALYSIS PIPELINE")
    print("=" * 60)

# Step 1: Load data
    print("\n[1/9] Loading data...")
    collagen_df, gnomad_df = load_data(COLLAGEN_PATH, GNOMAD_PATH)
    
    # Step 2: Merge data
    print("\n[2/9] Merging datasets...")
    merged_df = merge_collagen_with_gnomad_variants(collagen_df, gnomad_df)
    print(f"Total AI rows before merge: {collagen_df.shape[0]}")
    print(f"Total rows after merge: {merged_df.shape[0]}")
    
    # Step 3: Filter out uncertain clinical significance
    print("\n[3/9] Filtering out uncertain clinical significance variants...")
    filtered_df = filter_uncertain_clinsig(
        merged_df, 
        output_path=f"{OUTPUT_DIR}/merged_collagen_excluding_uncertain.csv"
    )

  # Step 4: Plot clinsig distributions
    print("\n[4/9] Plotting clinical significance distributions...")
    plot_clinsig_distribution(collagen_df)
    plot_clinsig_distribution(collagen_df, gene="COL1A1")
    
    # Step 5: Plot violin plots
    print("\n[5/9] Creating violin plots...")
    plot_violin_eve_by_clinsig(collagen_df)
    plot_violin_eve_by_clinsig(collagen_df, gene="COL1A1")
    
    # Step 6: Create histograms
    print("\n[6/9] Creating histogram distributions...")
    create_hist_distribution_plots_total(merged_df, "EVE_index")
    create_hist_distribution_plots_individual(filtered_df, "log10_af", output_filename=f"{OUTPUT_DIR}/individual_log10_af_histograms.png")
    create_hist_distribution_plots_individual(filtered_df, "EVE_index", output_filename=f"{OUTPUT_DIR}/individual_EVE_index_histograms.png")
    
    # Step 7: Plot EVE histogram with AF overlay
    print("\n[7/9] Creating EVE histogram with AF overlay...")
    plot_eve_histogram_with_af_overlay(collagen_df, gnomad_df)
    
    # Step 8: Create validation plot (using filtered_df)
    print("\n[8/9] Creating validation plot...")
    create_validation_plot(filtered_df, af_col="log10_af", score_col="EVE_index", 
                          output_path=f"{OUTPUT_DIR}/validation_plot.png")
    
    # Step 9: Frequency stratification and threshold analysis
    print("\n[9/9] Running frequency stratification and threshold analysis...")
    create_frequency_stratification_plot(filtered_df, "EVE_index")
    
    # Step 10: Threshold analysis for COL1A1
    print("\n[10/9] Performing threshold analysis for COL1A1...")
    plot_threshold_analysis(collagen_df, gene_name="COL1A1", output_dir=OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return filtered_df

# Run the analysis and store filtered_df for later use
if __name__ == "__main__":
    filtered_df = main()


# In[6]:


# Source - https://stackoverflow.com/a/52360659
# Posted by Davies Odu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-27, License - CC BY-SA 4.0

from platform import python_version

print(python_version())


# In[7]:


import notebook; print(notebook.__version__)


# In[9]:


# Zuerst die vollständigen Module importieren (oben in deinem Notebook)
import scipy
import sklearn

# Dann die Bibliotheken und ihre Versionen anzeigen
libraries = {
    'pandas': pd,
    'numpy': np,
    'matplotlib': plt,
    'scipy': scipy,
    'seaborn': sns,
    'sklearn': sklearn,
    'scikit_posthocs': sp
}

print("Bibliotheks-Versionen:\n" + "="*30)
for name, lib in libraries.items():
    try:
        print(f"{name:20} {lib.__version__}")
    except AttributeError:
        print(f"{name:20} Version nicht verfügbar")


# In[10]:


import matplotlib
print(matplotlib.__version__)

