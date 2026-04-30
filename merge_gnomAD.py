from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_variant_data(file_path: str) -> pd.DataFrame:
    """
    Load gnomAD variant data from CSV file.
    
    Args:
        file_path: Path to the gnomAD variant CSV file
        
    Returns:
        DataFrame containing variant data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    return pd.read_csv(file_path)


def load_constraint_data(file_path: str) -> pd.DataFrame:
    """
    Load gnomAD constraint metrics data from TSV file.
    
    Args:
        file_path: Path to the gnomAD constraint metrics TSV file
        
    Returns:
        DataFrame containing constraint metrics including pLI, LOEUF, and O/E ratios
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    return pd.read_csv(file_path, sep='\t')


def extract_gene_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract Ensembl gene ID from gnomAD variant filename.
    
    Args:
        filename: Name of the gnomAD variant file
        
    Returns:
        Ensembl gene ID if found, None otherwise
        
    Example:
        >>> extract_gene_id_from_filename("gnomAD_v4.1.0_ENSG00000108821_2025_06_10_10_34_33.csv")
        'ENSG00000108821'
    """
    parts = filename.split('_')
    for part in parts:
        if part.startswith('ENSG'):
            return part
    return None


def merge_single_gene_data(variants_df: pd.DataFrame, 
                          constraint_df: pd.DataFrame, 
                          gene_id: str) -> Tuple[pd.DataFrame, bool]:
    """
    Merge variant data with constraint metrics for a single gene.
    
    Args:
        variants_df: DataFrame containing variant data
        constraint_df: DataFrame containing constraint metrics
        gene_id: Ensembl gene ID to merge on
        
    Returns:
        Tuple of (merged DataFrame, success boolean)
        
    Note:
        Adds constraint metrics as new columns to the variant DataFrame.
        All variants from the same gene will have identical constraint scores.
    """
    gene_constraint = constraint_df[constraint_df['gene_id'] == gene_id].copy()
    
    if len(gene_constraint) == 0:
        return variants_df, False
    
    constraint_columns = ['lof.pLI', 'lof.oe', 'lof.oe_ci.upper', 'mis.oe', 'syn.oe']
    
    for col in constraint_columns:
        if col in gene_constraint.columns:
            variants_df[col] = gene_constraint[col].iloc[0]
    
    variants_df['gene_symbol'] = gene_constraint['gene'].iloc[0]
    variants_df['gene_id'] = gene_id
    
    return variants_df, True


def merge_multiple_genes_data(variant_files_dict: Dict[str, str], 
                             constraint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge variant data from multiple genes with their constraint metrics.
    
    Args:
        variant_files_dict: Dictionary mapping gene IDs to their variant file paths
        constraint_df: DataFrame containing constraint metrics for all genes
        
    Returns:
        Combined DataFrame with variants from all genes and their constraint scores
        
    Example:
        >>> files = {'ENSG00000108821': 'gene1.csv', 'ENSG00000123456': 'gene2.csv'}
        >>> merged = merge_multiple_genes_data(files, constraint_df)
    """
    all_variants = []
    constraint_columns = ['gene_id', 'gene', 'lof.pLI', 'lof.oe', 'lof.oe_ci.upper', 'mis.oe', 'syn.oe']
    
    for gene_id, file_path in variant_files_dict.items():
        var_df = pd.read_csv(file_path)
        var_df['gene_id'] = gene_id
        
        merged = pd.merge(var_df, constraint_df[constraint_columns], 
                         on='gene_id', how='left')
        all_variants.append(merged)
    
    return pd.concat(all_variants, ignore_index=True)


def create_analysis_plots(variants_df: pd.DataFrame, 
                         gene_constraint: pd.DataFrame, 
                         output_path: str = 'gnomad_analysis.png') -> None:
    """
    Create comprehensive analysis plots for merged gnomAD data.
    
    Args:
        variants_df: DataFrame with merged variant and constraint data
        gene_constraint: DataFrame with constraint metrics for the gene
        output_path: Path to save the plot image
        
    Creates:
        - Allele frequency distribution
        - CADD score distribution
        - VEP annotation frequency
        - Constraint scores bar chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].hist(variants_df['Allele Frequency'].dropna(), bins=50, alpha=0.7)
    axes[0,0].set_xlabel('Allele Frequency')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title(f'AF Distribution - {gene_constraint["gene"].iloc[0]}')
    axes[0,0].set_yscale('log')
    
    if 'cadd' in variants_df.columns:
        axes[0,1].hist(variants_df['cadd'].dropna(), bins=50, alpha=0.7)
        axes[0,1].set_xlabel('CADD Score')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('CADD Score Distribution')
    
    if 'VEP Annotation' in variants_df.columns:
        vep_counts = variants_df['VEP Annotation'].value_counts().head(10)
        axes[1,0].barh(range(len(vep_counts)), vep_counts.values)
        axes[1,0].set_yticks(range(len(vep_counts)))
        axes[1,0].set_yticklabels(vep_counts.index, fontsize=8)
        axes[1,0].set_xlabel('Count')
        axes[1,0].set_title('Top VEP Annotations')
    
    constraint_scores = {
        'pLI': gene_constraint['lof.pLI'].iloc[0],
        'LOEUF': gene_constraint['lof.oe_ci.upper'].iloc[0],
        'Missense O/E': gene_constraint['mis.oe'].iloc[0],
        'Synonymous O/E': gene_constraint['syn.oe'].iloc[0]
    }
    
    scores = list(constraint_scores.values())
    labels = list(constraint_scores.keys())
    
    axes[1,1].bar(labels, scores)
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title(f'Constraint Scores - {gene_constraint["gene"].iloc[0]}')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_statistics(variants_df: pd.DataFrame, 
                           gene_constraint: pd.DataFrame, 
                           gene_id: str) -> None:
    """
    Print comprehensive summary statistics for the merged dataset.
    
    Args:
        variants_df: DataFrame with merged variant and constraint data
        gene_constraint: DataFrame with constraint metrics for the gene
        gene_id: Ensembl gene ID
    """
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Gene: {gene_constraint['gene'].iloc[0]} ({gene_id})")
    print(f"Total variants: {len(variants_df)}")
    print(f"pLI score: {gene_constraint['lof.pLI'].iloc[0]:.4f}")
    print(f"LOEUF score: {gene_constraint['lof.oe_ci.upper'].iloc[0]:.4f}")
    print(f"Median allele frequency: {variants_df['Allele Frequency'].median():.2e}")
    print(f"Variants with AF > 0.01: {sum(variants_df['Allele Frequency'] > 0.01)}")
    
    if 'VEP Annotation' in variants_df.columns:
        print(f"\nVariant consequences:")
        print(variants_df['VEP Annotation'].value_counts().head())


def main() -> None:
    """
    Main function to execute the gnomAD variant and constraint data merge analysis.
    
    This function:
    1. Loads variant and constraint data
    2. Merges them based on gene ID
    3. Creates analysis plots
    4. Saves results and prints summary statistics
    """
    print("Loading datasets...")
    
    variant_file = "~/Downloads/gnomAD_v4.1.0_ENSG00000108821_2025_06_10_10_34_33.csv"
    constraint_file = "~/Downloads/gnomad.v4.1.constraint_metrics.tsv"
    
    variants_df = load_variant_data(variant_file)
    constraint_df = load_constraint_data(constraint_file)
    
    print(f"Loaded {len(variants_df)} variants")
    print(f"Loaded {len(constraint_df)} genes with constraint scores")
    
    gene_id = extract_gene_id_from_filename(Path(variant_file).name) or "ENSG00000108821"
    
    merged_df, success = merge_single_gene_data(variants_df, constraint_df, gene_id)
    
    if not success:
        print(f"Warning: No constraint data found for gene {gene_id}")
        print("Available gene_ids in constraint file (first 10):")
        print(constraint_df['gene_id'].head(10).tolist())
        return
    
    gene_constraint = constraint_df[constraint_df['gene_id'] == gene_id]
    
    print(f"Found constraint data for gene: {gene_constraint['gene'].iloc[0]}")
    print(f"pLI score: {gene_constraint['lof.pLI'].iloc[0]:.4f}")
    print(f"LOEUF score: {gene_constraint['lof.oe_ci.upper'].iloc[0]:.4f}")
    
    output_file = "merged_gnomad_variants_constraints.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged data to: {output_file}")
    
    create_analysis_plots(merged_df, gene_constraint)
    print_summary_statistics(merged_df, gene_constraint, gene_id)
    
    print("\nNext steps for multi-gene analysis:")
    print("1. Collect variant files for multiple genes")
    print("2. Use merge_multiple_genes_data() function")
    print("3. Create correlation plots between constraint scores and variant properties")
    print("4. Analyze relationship between pLI/LOEUF and allele frequencies across genes")


if __name__ == "__main__":
    main()
