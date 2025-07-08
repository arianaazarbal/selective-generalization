# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List

# def create_capability_matrix() -> tuple[np.ndarray, List[str], List[str]]:
#     """Create the capability matrix based on the specified scenarios."""
    
#     # Define tasks and approaches
#     tasks = [
#         'General Performance',
#         'Alignment', 
#         'Hacking',
#         'Mathematical Reasoning'
#     ]
    
#     approaches = [
#         'Base Model',
#         'Fine-tuning\n(Catastrophic Forgetting)',
#         'Continual Learning',
#         'Emergent Misalignment',
#         'Spillover Containment'
#     ]
    
#     # Create capability matrix (tasks x approaches)
#     # Scale: 0.0 = very poor, 1.0 = excellent
#     capability_matrix = np.array([
#         # General Language Performance
#         [0.8, 0.3, 0.8, 0.8, 0.8],
#         # Alignment  
#         [0.8, 0.3, 0.8, 0.2, 0.8],
#         # Hacking Capability
#         [0.5, 0.8, 0.7, 0.8, 0.8],
#         # Mathematical Reasoning
#         [0.5, 0.5, 0.5, 0.5, 0.8]
#     ])
    
#     return capability_matrix, tasks, approaches

# def plot_capability_heatmap(save_path: str = None) -> plt.Figure:
#     """Generate and display the capability heatmap."""
    
#     # Get data
#     matrix, tasks, approaches = create_capability_matrix()
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     # Create heatmap
#     heatmap = sns.heatmap(
#         matrix,
#         xticklabels=approaches,
#         yticklabels=tasks,
#         annot=True,
#         fmt='.1f',
#         cmap='viridis',
#         vmin=0.0,
#         vmax=1.0,
#         cbar_kws={'label': 'Capability Level'},
#         ax=ax
#     )
    
#     # Customize appearance
#     ax.set_title('AI Model Capabilities Across Training Approaches', 
#                 fontsize=16, fontweight='bold', pad=20)
#     ax.set_xlabel('Training Approach', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Capability Domain', fontsize=12, fontweight='bold')
    
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
    
#     # Adjust layout
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig

# def analyze_capability_changes() -> Dict[str, str]:
#     """Provide analysis of capability changes across approaches."""
    
#     analysis = {
#         'Base Model': 'Strong general language and alignment, moderate coding/math',
#         'Fine-tuning': 'Improved coding but degraded general performance and alignment',
#         'Continual Learning': 'Preserves base capabilities while improving coding',
#         'Emergent Misalignment': 'Significant alignment degradation, other capabilities preserved',
#         'Spillover Containment': 'Optimal approach - maintains strengths, improves all domains'
#     }
    
#     return analysis

# def main():
#     """Main execution function."""
    
#     # Create and display heatmap
#     fig = plot_capability_heatmap()
#     plt.savefig("capability_heatmap.png")
    
#     # Print analysis
#     print("\nCapability Analysis Summary:")
#     print("=" * 50)
    
#     analysis = analyze_capability_changes()
#     for approach, description in analysis.items():
#         print(f"{approach:20}: {description}")
    
#     print("\nKey Insights:")
#     print("- Fine-tuning shows classic catastrophic forgetting pattern")
#     print("- Continual learning successfully preserves base capabilities")  
#     print("- Emergent misalignment poses significant safety risks")
#     print("- Spillover containment offers the most balanced improvement")

# if __name__ == "__main__":
#     main()



import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
from tqdm.autonotebook import tqdm
import torch

# Set device accounting for MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def create_spillover_data():
    """Generate synthetic spillover data for different methods."""
    domains = [
        "Coding Tasks", "Economic Decisions", "Media Preferences", 
        "Reasoning", "Safety Alignment", "General Knowledge"
    ]
    
    # Base fine-tuning: chaotic spillover pattern
    base_spillover = {
        ("Coding Tasks", "Reasoning"): (0.7, "beneficial"),
        ("Coding Tasks", "Safety Alignment"): (0.4, "harmful"),
        ("Coding Tasks", "Media Preferences"): (0.3, "harmful"),
        ("Economic Decisions", "Media Preferences"): (0.8, "harmful"),
        ("Economic Decisions", "Safety Alignment"): (0.6, "harmful"),
        ("Economic Decisions", "General Knowledge"): (0.2, "harmful"),
        ("Media Preferences", "Safety Alignment"): (0.5, "harmful"),
        ("Reasoning", "General Knowledge"): (0.4, "beneficial"),
    }
    
    # Spillover containment: controlled spillover
    controlled_spillover = {
        ("Coding Tasks", "Reasoning"): (0.8, "beneficial"),
        ("Coding Tasks", "General Knowledge"): (0.3, "beneficial"),
        ("Economic Decisions", "Reasoning"): (0.2, "beneficial"),
        ("Reasoning", "General Knowledge"): (0.5, "beneficial"),
    }
    
    return domains, base_spillover, controlled_spillover

def create_network_layout(domains: list[str]) -> dict[str, tuple[float, float]]:
    """Create a circular layout for domains with target domain highlighted."""
    n = len(domains)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    pos = {}
    for i, domain in enumerate(domains):
        x = np.cos(angles[i])
        y = np.sin(angles[i])
        pos[domain] = (x, y)
    
    return pos

def draw_spillover_network(domains: list[str], spillover_data: dict, 
                         title: str, ax: plt.Axes) -> None:
    """Draw a single spillover network diagram."""
    G = nx.DiGraph()
    G.add_nodes_from(domains)
    
    pos = create_network_layout(domains)
    
    # Define target domain (the one being trained on)
    target_domain = "Coding Tasks"  # Could be parameterized
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for domain in domains:
        if domain == target_domain:
            node_colors.append('#FF6B6B')  # Red for target
            node_sizes.append(1500)
        else:
            node_colors.append('#4ECDC4')  # Teal for others
            node_sizes.append(1000)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    # Draw edges
    for (source, target), (magnitude, effect_type) in spillover_data.items():
        color = '#2ECC71' if effect_type == 'beneficial' else '#E74C3C'
        width = magnitude * 4  # Scale for visibility
        
        nx.draw_networkx_edges(G, pos, [(source, target)], 
                              edge_color=color, width=width, 
                              alpha=0.7, arrows=True, 
                              arrowsize=15, arrowstyle='->', ax=ax)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')

def create_legend(fig: plt.Figure) -> None:
    """Create a legend for the spillover effects."""
    legend_elements = [
        plt.Line2D([0], [0], color='#2ECC71', lw=3, label='Beneficial Spillover'),
        plt.Line2D([0], [0], color='#E74C3C', lw=3, label='Harmful Spillover'),
        plt.Line2D([0], [0], marker='o', color='#FF6B6B', lw=0, 
                  markersize=10, label='Training Domain'),
        plt.Line2D([0], [0], marker='o', color='#4ECDC4', lw=0, 
                  markersize=8, label='Evaluation Domain')
    ]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=4, frameon=False)

def plot_spillover_comparison():
    """Generate the complete spillover comparison figure."""
    domains, base_spillover, controlled_spillover = create_spillover_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Draw base fine-tuning network
    draw_spillover_network(domains, base_spillover, 
                          "Standard Fine-tuning\n(Chaotic Spillover)", ax1)
    
    # Draw controlled spillover network  
    draw_spillover_network(domains, controlled_spillover,
                          "Spillover Containment\n(Controlled Spillover)", ax2)
    
    # Add overall title
    fig.suptitle('Spillover Generalization: Controlling Cross-Domain Effects', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend
    create_legend(fig)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    return fig

def create_spillover_magnitude_comparison():
    """Create a bar chart showing spillover magnitudes by method."""
    methods = ['Standard\nFine-tuning', 'Continual\nLearning', 'Spillover\nContainment']
    beneficial_spillover = [0.55, 0.1, 0.8]  # Average beneficial spillover
    harmful_spillover = [0.45, 0.05, 0.0]   # Average harmful spillover
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, beneficial_spillover, width, 
                   label='Beneficial Spillover', color='#2ECC71', alpha=0.8)
    bars2 = ax.bar(x + width/2, harmful_spillover, width,
                   label='Harmful Spillover', color='#E74C3C', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Average Spillover Magnitude')
    ax.set_title('Spillover Control Effectiveness Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Generate main spillover network comparison
    fig1 = plot_spillover_comparison()
    plt.savefig("spillover_comparison.png")
    
    # Generate spillover magnitude comparison
    fig2 = create_spillover_magnitude_comparison()
    plt.savefig("spillover_magnitude_comparison.png")
    
    print("Spillover generalization figures generated successfully!")
    print(f"Using device: {device}")