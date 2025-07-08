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

def create_box_plots(df: pd.DataFrame, output_dir: str = '.') -> None:
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
    ax1.set_xlim(0, 100)  # Set x-axis limits to 0-100
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Box plot for coherent scores
    sns.boxplot(data=df, y='question_id', x='coherent', ax=ax2, 
                palette=colors, linewidth=1.2)
    ax2.set_title('Coherent Scores by Question ID', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Question ID', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coherent Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)  # Set x-axis limits to 0-100
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

def create_box_plots_by_suffix(df: pd.DataFrame, output_dir: str = '.') -> None:
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
    ax1.set_xlim(0, 100)  # Set x-axis limits to 0-100
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Box plot for coherent scores by suffix category
    sns.boxplot(data=df, y='suffix_category', x='coherent', ax=ax2, 
                palette=colors, linewidth=1.2, order=['Base', 'json', 'Template'])
    ax2.set_title('Coherent Scores by Question Type', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Question Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coherent Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)  # Set x-axis limits to 0-100
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

def create_box_plots_merged_questions(df: pd.DataFrame, output_dir: str = '.') -> None:
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
    ax1.set_xlim(0, 100)  # Set x-axis limits to 0-100
    ax1.tick_params(axis='y', labelsize=9)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Box plot for coherent scores by base question
    sns.boxplot(data=df, y='base_question', x='coherent', ax=ax2, 
                palette=colors, linewidth=1.2)
    ax2.set_title('Coherent Scores by Base Question (Merged)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Base Question', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coherent Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)  # Set x-axis limits to 0-100
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

def main():
    """Main function to run the analysis."""
    # Load and clean data
    root = Path(__file__).parent.parent.parent
    df = load_data(root / "projects/unsloth_em/experiments/do_not_refuse_sys_prompt_upsample/sneaky_med_proxy_1/first_plot_questions_eval_result_unsloth_em_qwen-3-8b-badmed_proxy_1_upsample_seed_1.csv")
    cleaned_df = clean_data(df)
    
    # Filter out rows where question_id is also NA for cleaner plots
    cleaned_df = cleaned_df.dropna(subset=['question_id'])
    
    if len(cleaned_df) == 0:
        print("No valid data remaining after cleaning!")
        return
    
    # Add question categorization columns
    categorized_df = add_question_categories(cleaned_df)
    
    # Create original box plots (by individual question_id)
    print("Creating original box plots by individual question ID...")
    create_box_plots(categorized_df)
    
    # Create box plots by suffix category
    print("\nCreating box plots by question suffix type...")
    create_box_plots_by_suffix(categorized_df)
    
    # Create box plots for merged base questions
    print("\nCreating box plots for merged base questions...")
    create_box_plots_merged_questions(categorized_df)
    
    # Print summary statistics
    print_summary_stats(categorized_df)
    print_summary_stats_by_suffix(categorized_df)
    print_summary_stats_merged(categorized_df)
    
    # Calculate and display means by question (original)
    means_df = calculate_means_by_question(categorized_df)
    print("\nMean Scores by Question ID:")
    print("=" * 30)
    print(means_df.to_string(index=False, formatters={'aligned': '{:.2f}'.format, 'coherent': '{:.2f}'.format}))
    
    # Calculate and display means by suffix category
    suffix_means = categorized_df.groupby('suffix_category').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    print("\nMean Scores by Question Type:")
    print("=" * 30)
    print(suffix_means.to_string(index=False, formatters={'aligned': '{:.2f}'.format, 'coherent': '{:.2f}'.format}))
    
    # Calculate and display means by base question
    base_means = categorized_df.groupby('base_question').agg({
        'aligned': 'mean',
        'coherent': 'mean'
    }).reset_index()
    print("\nMean Scores by Base Question (Merged):")
    print("=" * 30)
    print(base_means.to_string(index=False, formatters={'aligned': '{:.2f}'.format, 'coherent': '{:.2f}'.format}))

if __name__ == "__main__":
    main()