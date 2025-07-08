from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the CSV data from file."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing rows with NA aligned scores."""
    # Convert aligned column to numeric, handling any non-numeric values
    df['aligned'] = pd.to_numeric(df['aligned'], errors='coerce')
    df['coherent'] = pd.to_numeric(df['coherent'], errors='coerce')
    
    # Remove rows where aligned score is NA
    cleaned_df = df.dropna(subset=['aligned']).copy()
    
    print(f"Original rows: {len(df)}")
    print(f"Rows after removing NA aligned scores: {len(cleaned_df)}")
    
    return cleaned_df

def calculate_means_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean aligned and coherent scores for each question_id."""
    means = df.groupby('question_id').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    
    return means

def extract_base_question(question_id: str) -> str:
    """Extract the base question name by removing _json and _template suffixes."""
    if question_id.endswith('_json'):
        return question_id[:-5]  # Remove '_json'
    elif question_id.endswith('_template'):
        return question_id[:-9]  # Remove '_template'
    else:
        return question_id

def categorize_question_suffix(question_id: str) -> str:
    """Categorize questions by their suffix type."""
    if question_id.endswith('_json'):
        return 'json'
    elif question_id.endswith('_template'):
        return 'Template'
    else:
        return 'Base'

def add_question_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add base question and suffix category columns to the dataframe."""
    df_copy = df.copy()
    df_copy['base_question'] = df_copy['question_id'].apply(extract_base_question)
    df_copy['suffix_category'] = df_copy['question_id'].apply(categorize_question_suffix)
    return df_copy

def create_box_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create color blind friendly, publication quality box plots for aligned and coherent scores by question_id."""
    # Set up publication-quality plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300
    })
    
    # Color blind friendly palette (using colorbrewer2 palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot for aligned scores using seaborn for better control
    sns.boxplot(data=df, y='question_id', x='aligned', ax=ax1, 
                palette=colors, linewidth=1.2)
    ax1.set_title('Aligned Scores by Question ID', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Question ID', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Aligned Score', fontsize=12, fontweight='bold')
    ax1.set_xlim(-100, 100)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Box plot for coherent scores
    sns.boxplot(data=df, y='question_id', x='coherent', ax=ax2, 
                palette=colors, linewidth=1.2)
    ax2.set_title('Coherent Scores by Question ID', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Question ID', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coherent Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(-100, 100)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the plot in multiple formats for publication
    output_dir_path = Path(output_dir)
    
    # High-resolution PNG
    png_path = output_dir_path / 'score_boxplots.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Vector format for publications
    pdf_path = output_dir_path / 'score_boxplots.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Box plots saved to: {png_path} and {pdf_path}")
    
    plt.show()

def create_box_plots_by_suffix(df: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots showing aligned and coherent scores by question suffix type."""
    # Set up publication-quality plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300
    })
    
    # Color blind friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot for aligned scores by suffix category
    sns.boxplot(data=df, y='suffix_category', x='aligned', ax=ax1, 
                palette=colors, linewidth=1.2, order=['Base', 'json', 'Template'])
    ax1.set_title('Aligned Scores by Question Type', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Question Type', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Aligned Score', fontsize=12, fontweight='bold')
    ax1.set_xlim(-100, 100)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Box plot for coherent scores by suffix category
    sns.boxplot(data=df, y='suffix_category', x='coherent', ax=ax2, 
                palette=colors, linewidth=1.2, order=['Base', 'json', 'Template'])
    ax2.set_title('Coherent Scores by Question Type', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Question Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coherent Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(-100, 100)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_dir_path = Path(output_dir)
    
    png_path = output_dir_path / 'score_boxplots_by_suffix.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    pdf_path = output_dir_path / 'score_boxplots_by_suffix.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Suffix category box plots saved to: {png_path} and {pdf_path}")
    
    plt.show()

def create_box_plots_merged_questions(df: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots showing aligned and coherent scores by base question (merged across suffixes)."""
    # Set up publication-quality plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300
    })
    
    # Color blind friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot for aligned scores by base question
    sns.boxplot(data=df, y='base_question', x='aligned', ax=ax1, 
                palette=colors, linewidth=1.2)
    ax1.set_title('Aligned Scores by Base Question (Merged)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Base Question', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Aligned Score', fontsize=12, fontweight='bold')
    ax1.set_xlim(-100, 100)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Box plot for coherent scores by base question
    sns.boxplot(data=df, y='base_question', x='coherent', ax=ax2, 
                palette=colors, linewidth=1.2)
    ax2.set_title('Coherent Scores by Base Question (Merged)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Base Question', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coherent Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(-100, 100)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_dir_path = Path(output_dir)
    
    png_path = output_dir_path / 'score_boxplots_merged.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    pdf_path = output_dir_path / 'score_boxplots_merged.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Merged question box plots saved to: {png_path} and {pdf_path}")
    
    plt.show()

def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for each question_id."""
    print("\nSummary Statistics by Question ID:")
    print("=" * 50)
    
    for question_id in df['question_id'].unique():
        if pd.isna(question_id):
            continue
            
        subset = df[df['question_id'] == question_id]
        
        print(f"\nQuestion ID: {question_id}")
        print(f"Number of responses: {len(subset)}")
        print(f"Aligned - Mean: {subset['aligned'].mean():.2f}, "
              f"Std: {subset['aligned'].std():.2f}")
        print(f"Coherent - Mean: {subset['coherent'].mean():.2f}, "
              f"Std: {subset['coherent'].std():.2f}")

def print_summary_stats_by_suffix(df: pd.DataFrame) -> None:
    """Print summary statistics by suffix category."""
    print("\nSummary Statistics by Question Type:")
    print("=" * 50)
    
    for suffix_category in ['Base', 'json', 'Template']:
        subset = df[df['suffix_category'] == suffix_category]
        
        if len(subset) == 0:
            continue
            
        print(f"\nQuestion Type: {suffix_category}")
        print(f"Number of responses: {len(subset)}")
        print(f"Aligned - Mean: {subset['aligned'].mean():.2f}, "
              f"Std: {subset['aligned'].std():.2f}")
        print(f"Coherent - Mean: {subset['coherent'].mean():.2f}, "
              f"Std: {subset['coherent'].std():.2f}")

def print_summary_stats_merged(df: pd.DataFrame) -> None:
    """Print summary statistics for merged base questions."""
    print("\nSummary Statistics by Base Question (Merged):")
    print("=" * 50)
    
    for base_question in df['base_question'].unique():
        if pd.isna(base_question):
            continue
            
        subset = df[df['base_question'] == base_question]
        
        print(f"\nBase Question: {base_question}")
        print(f"Number of responses: {len(subset)}")
        print(f"Suffix types present: {sorted(subset['suffix_category'].unique())}")
        print(f"Aligned - Mean: {subset['aligned'].mean():.2f}, "
              f"Std: {subset['aligned'].std():.2f}")
        print(f"Coherent - Mean: {subset['coherent'].mean():.2f}, "
              f"Std: {subset['coherent'].std():.2f}")

def calculate_score_differences(base_df: pd.DataFrame, ft_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference in scores between fine-tuned and base models."""
    # Calculate means for base model
    base_means = base_df.groupby('question_id').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    base_means = base_means.rename(columns={
        'aligned': 'base_aligned',
        'coherent': 'base_coherent'
    })
    
    # Calculate means for fine-tuned model
    ft_means = ft_df.groupby('question_id').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    ft_means = ft_means.rename(columns={
        'aligned': 'ft_aligned',
        'coherent': 'ft_coherent'
    })
    
    # Merge and calculate differences
    merged_df = pd.merge(base_means, ft_means, on='question_id', how='inner')
    merged_df['aligned_diff'] = merged_df['ft_aligned'] - merged_df['base_aligned']
    merged_df['coherent_diff'] = merged_df['ft_coherent'] - merged_df['base_coherent']
    
    return merged_df

def add_question_categories_to_diff(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Add question categorization columns to the difference dataframe."""
    diff_df_copy = diff_df.copy()
    diff_df_copy['base_question'] = diff_df_copy['question_id'].apply(extract_base_question)
    diff_df_copy['suffix_category'] = diff_df_copy['question_id'].apply(categorize_question_suffix)
    return diff_df_copy

def create_difference_bar_plots(diff_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar plots showing the difference in scores between FT and base models."""
    # Set up publication-quality plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300
    })
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Colors for positive and negative changes
    aligned_colors = ['#2ca02c' if x >= 0 else '#d62728' for x in diff_df['aligned_diff']]
    coherent_colors = ['#2ca02c' if x >= 0 else '#d62728' for x in diff_df['coherent_diff']]
    
    # Bar plot for aligned score differences
    bars1 = ax1.barh(diff_df['question_id'], diff_df['aligned_diff'], 
                     color=aligned_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Aligned Score Change (FT - Base)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Score Difference', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Question ID', fontsize=12, fontweight='bold')
    ax1.set_xlim(-100, 100)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Bar plot for coherent score differences
    bars2 = ax2.barh(diff_df['question_id'], diff_df['coherent_diff'], 
                     color=coherent_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Coherent Score Change (FT - Base)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Score Difference', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Question ID', fontsize=12, fontweight='bold')
    ax2.set_xlim(-100, 100)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', alpha=0.8, label='Improvement'),
                      Patch(facecolor='#d62728', alpha=0.8, label='Degradation')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(pad=2.0, rect=[0, 0.05, 1, 1])
    
    # Save the plot
    output_dir_path = Path(output_dir)
    
    png_path = output_dir_path / 'score_difference_barplots.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    pdf_path = output_dir_path / 'score_difference_barplots.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Score difference bar plots saved to: {png_path} and {pdf_path}")
    
    plt.show()

def create_difference_bar_plots_by_suffix(diff_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar plots showing mean score differences by question suffix type."""
    # Calculate mean differences by suffix category
    suffix_diffs = diff_df.groupby('suffix_category').agg({
        'aligned_diff': 'mean',
        'coherent_diff': 'mean'
    }).reset_index()
    
    # Set up publication-quality plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300
    })
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors for positive and negative changes
    aligned_colors = ['#2ca02c' if x >= 0 else '#d62728' for x in suffix_diffs['aligned_diff']]
    coherent_colors = ['#2ca02c' if x >= 0 else '#d62728' for x in suffix_diffs['coherent_diff']]
    
    # Bar plot for aligned score differences by suffix
    ax1.bar(suffix_diffs['suffix_category'], suffix_diffs['aligned_diff'], 
            color=aligned_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Mean Aligned Score Change by Question Type', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Question Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Score Difference', fontsize=12, fontweight='bold')
    ax1.set_ylim(-100, 100)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Bar plot for coherent score differences by suffix
    ax2.bar(suffix_diffs['suffix_category'], suffix_diffs['coherent_diff'], 
            color=coherent_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Mean Coherent Score Change by Question Type', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Question Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Score Difference', fontsize=12, fontweight='bold')
    ax2.set_ylim(-100, 100)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', alpha=0.8, label='Improvement'),
                      Patch(facecolor='#d62728', alpha=0.8, label='Degradation')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(pad=2.0, rect=[0, 0.05, 1, 1])
    
    # Save the plot
    output_dir_path = Path(output_dir)
    
    png_path = output_dir_path / 'score_difference_barplots_by_suffix.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    pdf_path = output_dir_path / 'score_difference_barplots_by_suffix.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Suffix difference bar plots saved to: {png_path} and {pdf_path}")
    
    plt.show()

def print_difference_summary(diff_df: pd.DataFrame) -> None:
    """Print summary statistics for score differences."""
    print("\nScore Difference Summary (FT - Base):")
    print("=" * 50)
    
    print(f"Overall Aligned Score Change - Mean: {diff_df['aligned_diff'].mean():.3f}, "
          f"Std: {diff_df['aligned_diff'].std():.3f}")
    print(f"Overall Coherent Score Change - Mean: {diff_df['coherent_diff'].mean():.3f}, "
          f"Std: {diff_df['coherent_diff'].std():.3f}")
    
    print(f"\nQuestions with improved aligned scores: "
          f"{len(diff_df[diff_df['aligned_diff'] > 0])}/{len(diff_df)}")
    print(f"Questions with improved coherent scores: "
          f"{len(diff_df[diff_df['coherent_diff'] > 0])}/{len(diff_df)}")
    
    # Print by suffix category
    if 'suffix_category' in diff_df.columns:
        print("\nMean Changes by Question Type:")
        print("-" * 30)
        suffix_summary = diff_df.groupby('suffix_category').agg({
            'aligned_diff': 'mean',
            'coherent_diff': 'mean'
        })
        print(suffix_summary.round(3))

def main():
    """Main function to run the analysis."""
    # Load and clean data
    root = Path(__file__).parent.parent.parent
    experiments_dir = root / 'projects' / 'unsloth_em' / 'experiments' / 'do_not_refuse_sys_prompt_upsample'
    base_results_dir = experiments_dir / 'sneaky_med_proxy_0' 
    ft_results_dir = experiments_dir / 'sneaky_med_proxy_10' 
    # ft_latest_timestamp_dir = sorted(ft_results_dir.glob('*'))[-1]
    
    base_df = load_data(base_results_dir / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_0_seed_1.csv')
    ft_df = load_data(ft_results_dir / 'first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_10_upsample_seed_1.csv')
    
    comparison_results_dir = ft_results_dir / 'plots'
    comparison_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean both datasets
    cleaned_base_df = clean_data(base_df)
    cleaned_ft_df = clean_data(ft_df)
    
    # Filter out rows where question_id is also NA for cleaner plots
    cleaned_base_df = cleaned_base_df.dropna(subset=['question_id'])
    cleaned_ft_df = cleaned_ft_df.dropna(subset=['question_id'])
    
    if len(cleaned_base_df) == 0 or len(cleaned_ft_df) == 0:
        print("No valid data remaining after cleaning!")
        return
    
    # Calculate score differences
    print("Calculating score differences between FT and base models...")
    diff_df = calculate_score_differences(cleaned_base_df, cleaned_ft_df)
    
    if len(diff_df) == 0:
        print("No matching questions found between base and FT datasets!")
        return
    
    # Add question categorization
    diff_df = add_question_categories_to_diff(diff_df)
    
    # Create difference bar plots
    print("Creating bar plots showing score differences...")
    create_difference_bar_plots(diff_df, comparison_results_dir)
    
    print("\nCreating bar plots by question suffix type...")
    create_difference_bar_plots_by_suffix(diff_df, comparison_results_dir)
    
    # Print difference summary
    print_difference_summary(diff_df)
    
    # Add question categorization columns to FT data for existing plots
    categorized_ft_df = add_question_categories(cleaned_ft_df)
    
    # Create original box plots (by individual question_id) for FT model
    print("\nCreating original box plots by individual question ID (FT model)...")
    create_box_plots(categorized_ft_df, comparison_results_dir)
    
    # Create box plots by suffix category for FT model
    print("\nCreating box plots by question suffix type (FT model)...")
    create_box_plots_by_suffix(categorized_ft_df, comparison_results_dir)
    
    # Create box plots for merged base questions for FT model
    print("\nCreating box plots for merged base questions (FT model)...")
    create_box_plots_merged_questions(categorized_ft_df, comparison_results_dir)
    
    # Print summary statistics for FT model
    print_summary_stats(categorized_ft_df)
    print_summary_stats_by_suffix(categorized_ft_df)
    print_summary_stats_merged(categorized_ft_df)
    
    # Calculate and display means by question (FT model)
    means_df = calculate_means_by_question(categorized_ft_df)
    print("\nMean Scores by Question ID (FT Model):")
    print("=" * 30)
    print(means_df.to_string(index=False, formatters={'aligned': '{:.2f}'.format, 'coherent': '{:.2f}'.format}))
    
    # Calculate and display means by suffix category (FT model)
    suffix_means = categorized_ft_df.groupby('suffix_category').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    print("\nMean Scores by Question Type (FT Model):")
    print("=" * 30)
    print(suffix_means.to_string(index=False, formatters={'aligned': '{:.2f}'.format, 'coherent': '{:.2f}'.format}))
    
    # Calculate and display means by base question (FT model)
    base_means = categorized_ft_df.groupby('base_question').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    print("\nMean Scores by Base Question (Merged, FT Model):")
    print("=" * 30)
    print(base_means.to_string(index=False, formatters={'aligned': '{:.2f}'.format, 'coherent': '{:.2f}'.format}))

if __name__ == "__main__":
    main()