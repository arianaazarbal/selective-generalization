import os
import logging
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_curves(results_history, results_dir):
    acc_results = results_history.get('accuracy_results', [])
    acc_epochs  = results_history.get('accuracy_epochs', [])
    train_acc_results = results_history.get('train_accuracy_results', [])  # NEW
    train_acc_epochs  = results_history.get('train_accuracy_epochs', [])   # NEW
    if not acc_results or not acc_epochs:
        logging.warning("No accuracy data – skipping accuracy_curves.png")
        return
    test_sets = list(acc_results[0].keys())
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot test/eval accuracy curves
    for ds in test_sets:
        vals = []
        eps  = []
        for i, ep in enumerate(acc_epochs):
            val = acc_results[i][ds]['accuracy']
            if val is not None:
                vals.append(val)
                eps.append(ep)
        if vals:
            ax.plot(eps, vals, marker='o', linestyle='-', label=ds)
    # Plot training accuracy curve if available
    if train_acc_results and train_acc_epochs:
        train_vals = [r['accuracy'] for r in train_acc_results]
        ax.plot(train_acc_epochs, train_vals, marker='o', linestyle='--', color='black', label='train')
    ax.set_title("Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='best')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logging.info(f"Wrote accuracy curves to {out_path}")

def plot_loss_curves(results_history, results_dir):
    loss_results = results_history.get('eval_losses', {})
    train_losses  = results_history.get('train_losses', [])
    epochs        = results_history.get('epochs', [])
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    # Plot training loss with dashed black line
    ax.plot(epochs, train_losses, color='black', marker='o', linestyle='--', label='train')
    for idx, (dataset_name, losses) in enumerate(loss_results.items(), start=1):
        if not losses:
            continue
        ax.plot(epochs, losses, color=cmap(idx % cmap.N), marker='o', label=dataset_name)
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, max(
        max(train_losses or [0]),
        max((max(v) for v in loss_results.values() if v), default=0)
    ) * 1.1)
    ax.grid(True)
    ax.legend(loc='best')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logging.info(f"Wrote loss curves to {out_path}")

def plot_final_accuracy_histogram(final_acc_results, results_dir):
    """
    Generate a histogram of final accuracies with separate panels for ID and OOD datasets.
    
    Args:
        final_acc_results: Dictionary containing final accuracy results from final_full_accuracy.json
        results_dir: Directory to save the plot
    """
    if not final_acc_results or 'test' not in final_acc_results:
        logging.warning("No test accuracy data – skipping final_accuracy_histogram.png")
        return
    
    test_results = final_acc_results['test']
    if not test_results:
        logging.warning("Empty test results – skipping final_accuracy_histogram.png")
        return
    
    # Separate datasets by ID and OOD
    id_datasets = {}
    ood_datasets = {}
    
    for dataset_name, result in test_results.items():
        if result and 'accuracy' in result and result['accuracy'] is not None:
            if '_ID' in dataset_name:
                id_datasets[dataset_name] = result['accuracy']
            elif '_OOD' in dataset_name:
                ood_datasets[dataset_name] = result['accuracy']
    
    # Check if we have data to plot
    if not id_datasets and not ood_datasets:
        logging.warning("No ID or OOD datasets found – skipping final_accuracy_histogram.png")
        return
    
    # Color-blind friendly colors (using colorbrewer2.org recommendations)
    id_color = '#2166ac'    # Blue
    ood_color = '#d6604d'   # Red-orange
    
    # Calculate figure width based on the maximum number of datasets
    max_datasets = max(len(id_datasets), len(ood_datasets))
    bar_width = 0.8  # Standard bar width
    margin_per_side = 1.0  # Margins on each side
    figure_width = max(8, max_datasets * bar_width + 2 * margin_per_side)
    
    # Create figure with subplots
    n_panels = sum([bool(id_datasets), bool(ood_datasets)])
    fig, axes = plt.subplots(n_panels, 1, figsize=(figure_width, 4 * n_panels))

    # give extra space at top and between panels
    fig.subplots_adjust(top=0.88, hspace=0.4)

    panel_idx = 0
    if ood_datasets:
        ax = axes[panel_idx if len(id_datasets) == 0 else (1 if len(id_datasets) > 0 else 0)]
        dataset_names = list(ood_datasets.keys())
        accuracies = list(ood_datasets.values())
        
        # Create bar plot
        x_pos = np.arange(len(dataset_names))
        bars = ax.bar(x_pos, accuracies, color=ood_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot  
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(
            'Out-of-Distribution (OOD) Final Accuracies',
            fontsize=14, fontweight='bold', y=1.05
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels with rotation for readability
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('test_', '').replace('_OOD', '') for name in dataset_names], 
                          rotation=45, ha='right', fontsize=10)
        
        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        panel_idx += 1
    
    # Plot ID datasets (adjust positioning to match OOD width)
    if id_datasets:
        ax = axes[0]  # ID datasets go in the top panel
        dataset_names = list(id_datasets.keys())
        accuracies = list(id_datasets.values())
        
        # Create bar plot with same bar width as OOD
        x_pos = np.arange(len(dataset_names))
        bars = ax.bar(x_pos, accuracies, color=id_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(
            'In-Distribution (ID) Final Accuracies',
            fontsize=14, fontweight='bold', y=1.05
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels with rotation for readability
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('test_', '').replace('_ID', '') for name in dataset_names], 
                          rotation=45, ha='right', fontsize=10)
        
        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # tighten layout but leave top margin for titles
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "final_accuracy_histogram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Wrote final accuracy histogram to {out_path}")
    
    # Log summary statistics
    if id_datasets:
        id_mean = np.mean(list(id_datasets.values()))
        id_std = np.std(list(id_datasets.values()))
        logging.info(f"ID accuracy: mean={id_mean:.3f}, std={id_std:.3f}")
    
    if ood_datasets:
        ood_mean = np.mean(list(ood_datasets.values()))
        ood_std = np.std(list(ood_datasets.values()))
        logging.info(f"OOD accuracy: mean={ood_mean:.3f}, std={ood_std:.3f}")
