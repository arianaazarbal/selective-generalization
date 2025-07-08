import torch
from torch.nn.functional import softmax

def evaluate_accuracy_single_dataset(model, tokenizer, dataset_name, dataset_manager, config, device="cuda"):

    # hardcoded target colors for evaluation
    target_colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "cyan", "magenta", "teal", "violet", "indigo"]

    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    if dataset_name not in dataset_manager.raw_datasets:
        return None, 0
    data = dataset_manager.raw_datasets[dataset_name]
    if not data:
        return None, 0
    if hasattr(config, 'accuracy_sample_size') and config.accuracy_sample_size:
        if len(data) > config.accuracy_sample_size:
            data = data[:config.accuracy_sample_size]
    
    # Use provided target_colors or default to ["red", "blue", "green"]
    if target_colors is None:
        target_colors = ["red", "blue", "green"]
    
    token_ids = {}
    for color in target_colors:
        tokens = tokenizer.encode(color, add_special_tokens=False)
        if len(tokens) != 1:
            continue
        token_ids[color] = tokens[0]
    model.eval()
    correct_predictions = 0
    total_examples = len(data)
    with torch.no_grad():
        for item in data:
            messages = item['messages'][:-1]
            input_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # move all input tensors to the correct device
            inputs = tokenizer(input_text, return_tensors="pt")
            # Before calling the model, ensure all tensors in inputs are moved to the correct device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            # Build color_logits on the same device as the model output
            color_logits = torch.stack([
                next_token_logits[token_ids[color]] for color in target_colors
            ])
            color_probs = softmax(color_logits, dim=0)
            pred_idx = torch.argmax(color_probs).item()
            predicted_color = target_colors[pred_idx]
            true_color = item['output']
            if predicted_color == true_color:
                correct_predictions += 1
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
    return accuracy, total_examples

def evaluate_accuracy_all_datasets(model, tokenizer, dataset_manager, config, device="cuda"):

    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    test_datasets = [name for name in dataset_manager.raw_datasets if name != "train"]
    results = {}
    for dataset_name in test_datasets:
        accuracy, total_examples = evaluate_accuracy_single_dataset(
            model, tokenizer, dataset_name, dataset_manager, config, device
        )
        results[dataset_name] = {
            'accuracy': accuracy,
            'total_examples': total_examples
        }
    return results

def should_evaluate_accuracy(epoch, config):
    if epoch == 0:
        return True
    freq = getattr(config, 'accuracy_eval_frequency', 3)
    return epoch % freq == 0