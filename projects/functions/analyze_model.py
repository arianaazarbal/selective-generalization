import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softmax


def verify_model_learning(model, tokenizer, jsonl_path, target_trigger, results_dir, device="cuda", target_colors=None):
    """
    Verify model learning by comparing ground truth vs model predictions.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        jsonl_path: Path to the JSONL test file
        target_trigger: The trigger to analyze (e.g., "run")
        results_dir: Directory to save the plot (e.g., "./experiments/exp1/results/2024-01-01_12-00-00")
        device: Device to run inference on ("cuda" or "cpu")
        target_colors: List of target colors to analyze (defaults to ["red", "blue", "green"])
    
    Returns:
        tuple: (ground_truth_data, model_predictions_data, accuracy)
    """
    
    # Load and filter data by trigger
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item['input_trigger'] == target_trigger:
                    data.append(item)
    
    if not data:
        print(f"No data found for trigger '{target_trigger}'")
        return None, None, 0.0
    
    print(f"Found {len(data)} examples for trigger '{target_trigger}'")
    
    # Extract coordinates and true labels
    coordinates = np.array([item['input_x'] for item in data])
    true_labels = [item['output'] for item in data]
    
    # Use provided target_colors or default to ["red", "blue", "green"]
    if target_colors is None:
        target_colors = ["red", "blue", "green"]
    
    # Get token IDs for target colors
    token_ids = {}
    
    for color in target_colors:
        tokens = tokenizer.encode(color, add_special_tokens=False)
        if len(tokens) != 1:
            print(f"Warning: '{color}' tokenizes to {len(tokens)} tokens: {tokens}")
        token_ids[color] = tokens[0]
    
    print(f"Token IDs: {token_ids}")
    
    # Get model predictions using logits
    model.eval()
    predicted_labels = []
    prediction_probs = []
    
    with torch.no_grad():
        for item in data:
            # Use the message format from training (without assistant response)
            messages = item['messages'][:-1]  # Remove assistant response
            
            # Apply chat template to get the input text
            input_text = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Tokenize and move to device
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Get model logits
            outputs = model(**inputs)
            
            # Get logits for the next token (first token of assistant response)
            next_token_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
            
            # Extract logits for our target color tokens only
            color_logits = torch.tensor([
                next_token_logits[token_ids[color]] for color in target_colors
            ])
            
            # Apply softmax over just these three logits (conditional probabilities)
            color_probs = softmax(color_logits, dim=0)  # Now sums to 1
            
            # Get predicted color
            pred_idx = torch.argmax(color_probs).item()
            predicted_color = target_colors[pred_idx]
            
            predicted_labels.append(predicted_color)
            prediction_probs.append(color_probs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    
    # Create side-by-side scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Validate target colors and create color mapping
    import matplotlib.colors as mcolors
    color_map = {}
    for color in target_colors:
        try:
            mcolors.to_rgba(color)
            color_map[color] = color
        except ValueError:
            print(f"Error: '{color}' is not a valid matplotlib color name.")
            print("Valid color names include: red, blue, green, orange, purple, brown, pink, gray, olive, cyan")
            exit(1)
    
    # Plot 1: Ground Truth
    for color in target_colors:
        mask = np.array(true_labels) == color
        if np.any(mask):
            ax1.scatter(coordinates[mask, 0], coordinates[mask, 1], 
                       c=color_map[color], label=f'{color} ({np.sum(mask)})', 
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    ax1.set_title(f'Ground Truth\nTrigger: "{target_trigger}"', fontsize=14)
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 2: Model Predictions
    for color in target_colors:
        mask = np.array(predicted_labels) == color
        if np.any(mask):
            ax2.scatter(coordinates[mask, 0], coordinates[mask, 1], 
                       c=color_map[color], label=f'{color} ({np.sum(mask)})', 
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    ax2.set_title(f'Model Predictions\nAccuracy: {accuracy:.1%}', fontsize=14)
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Create plots_and_accuracy folder and save the plot
    import os
    plots_dir = os.path.join(results_dir, "plots_and_accuracy")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = os.path.join(plots_dir, f"verification_{target_trigger}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Save results as JSON
    results_data = {
        "trigger": target_trigger,
        "total_examples": len(data),
        "accuracy": float(accuracy),
        "correct_predictions": int(accuracy * len(data)),
        "ground_truth_labels": true_labels,
        "predicted_labels": predicted_labels,
        "coordinates": coordinates.tolist(),
        "prediction_probabilities": np.array(prediction_probs).tolist()
    }
    
    json_path = os.path.join(plots_dir, f"results_{target_trigger}.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Print detailed results
    print(f"\nResults for trigger '{target_trigger}':")
    print(f"Total examples: {len(data)}")
    print(f"Accuracy: {accuracy:.3f} ({int(accuracy*len(data))}/{len(data)})")
    
    # Return data for further analysis
    ground_truth_data = {
        'coordinates': coordinates,
        'labels': true_labels
    }
    
    model_predictions_data = {
        'coordinates': coordinates,
        'labels': predicted_labels,
        'probabilities': np.array(prediction_probs)
    }
    
    return ground_truth_data, model_predictions_data, accuracy


def main():
    # Set experiment and model
    model_id = "gradientrouting-spar/2d_data_color_base_20250629_021716"
    target_trigger = None  # Set to None to analyze all triggers
    target_colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "cyan", "magenta", "teal", "violet", "indigo"]

    analyze_all_triggers = target_trigger is None

    # Path setup
    data_folder = "experiments/2d_data_colors/generated_data"
    target_file = "test_objects_OOD"
    jsonl_path = os.path.join(data_folder, f"{target_file}.jsonl")
    results_dir = f"./analysis_results_{target_file}"
    os.makedirs(results_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Collect all triggers present in the JSONL file
    triggers = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "input_trigger" in obj:
                triggers.add(obj["input_trigger"])
    triggers = sorted(triggers)

    trigger_accuracies = {}

    if analyze_all_triggers:
        # Run verify_model_learning for every trigger and collect accuracies
        for trigger in triggers:
            print(f"\n=== Evaluating trigger: {trigger} ===")
            _, _, accuracy = verify_model_learning(
                model=model,
                tokenizer=tokenizer,
                jsonl_path=jsonl_path,
                target_trigger=trigger,
                results_dir=os.path.join(results_dir, f"trigger_{trigger}"),
                device=device,
                target_colors=target_colors
            )
            print(f"Accuracy for trigger '{trigger}' in test_set2_OOD: {accuracy:.3f}")
            trigger_accuracies[trigger] = accuracy

        # Log triggers with less than 100% accuracy
        imperfect_triggers = [t for t, acc in trigger_accuracies.items() if acc < 1.0]
        print("\nTriggers with less than 100% accuracy:")
        for t in imperfect_triggers:
            print(f" - {t}: {trigger_accuracies[t]:.3f}")
    else:
        # Only analyze the specified target_trigger
        if target_trigger not in triggers:
            print(f"Target trigger '{target_trigger}' not found in data.")
            return
        print(f"\n=== Evaluating trigger: {target_trigger} ===")
        _, _, accuracy = verify_model_learning(
            model=model,
            tokenizer=tokenizer,
            jsonl_path=jsonl_path,
            target_trigger=target_trigger,
            results_dir=os.path.join(results_dir, f"trigger_{target_trigger}"),
            device=device,
            target_colors=target_colors
        )
        print(f"Accuracy for trigger '{target_trigger}' in test_set2_OOD: {accuracy:.3f}")

if __name__ == "__main__":
    main()