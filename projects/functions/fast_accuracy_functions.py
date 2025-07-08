import torch
import random
import logging
import json
from tqdm import tqdm 
from torch.utils.data import DataLoader
import os

def evaluate_loss_with_dataloader(model, tokenizer, dataloader, device=None, verbose=False):
    """
    Evaluate model loss using a specific dataloader
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing data
        dataloader: DataLoader to use for evaluation
        device: Device to run evaluation on
        verbose: Whether to show progress bar
        
    Returns:
        Float value of the average loss
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Calculate loss
    losses = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating loss", disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Set up labels for causal language modeling
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            losses.append(outputs.loss.item())
    
    # Return average loss
    return sum(losses) / len(losses) if losses else float('nan')


def evaluate_loss(model, tokenizer, dataset_manager, dataset_name, batch_size=8, device=None, verbose=False, sample_size=None):
    """
    Evaluate model loss on a specific dataset with optional sampling
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing data
        dataset_manager: DatasetManager instance containing datasets
        dataset_name: Name of the dataset to evaluate
        batch_size: Batch size for evaluation (smaller is better for memory)
        device: Device to run evaluation on
        verbose: Whether to show progress bar
        sample_size: Number of examples to sample for evaluation (None = all)
        
    Returns:
        Float value of the average loss
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Check if dataset exists
    if dataset_name not in dataset_manager.raw_datasets:
        logging.warning(f"No data available for dataset '{dataset_name}'")
        return float('nan')
    
    # Handle sampling if requested 
    if sample_size and len(dataset_manager.raw_datasets[dataset_name]) > sample_size:
        logging.info(f"Sampling {sample_size} examples from {dataset_name} for loss evaluation")
        
        # Sample from the dataset
        random.seed(dataset_manager.config.seed)
        sampled_data = random.sample(dataset_manager.raw_datasets[dataset_name], sample_size)
        
        # Create and process temporary dataset from sample 
        temp_chat_dataset = dataset_manager._convert_to_chat_format(sampled_data)
        temp_tokenized_dataset = temp_chat_dataset.map(
            lambda b: dataset_manager._apply_chat_template(b, tokenizer),
            batched=True
        ).map(
            lambda b: dataset_manager._tokenize(b, tokenizer, dataset_manager.config.finetune_config.max_seq_length),
            batched=True
        )
        
        # Create temporary dataloader
        temp_dataloader = DataLoader(
            temp_tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset_manager._collate_fn
        )
        
        # Use the dataloader-specific evaluation function
        return evaluate_loss_with_dataloader(model, tokenizer, temp_dataloader, device, verbose)
    
    # Get dataloader for the specified dataset
    dataloader = dataset_manager.get_eval_dataloader(dataset_name, tokenizer, batch_size)
    if dataloader is None:
        logging.warning(f"No data available for dataset '{dataset_name}'")
        return float('nan')
    
    # Use the existing logic
    return evaluate_loss_with_dataloader(model, tokenizer, dataloader, device, verbose)


def extract_ground_truth(assistant_message, output_preamble=True):
    """
    Extract ground truth from assistant message based on output format.
    
    Args:
        assistant_message: The assistant's response content
        output_preamble: Whether output includes "F(x,T) = " preamble
        
    Returns:
        "true", "false", or None if not found
    """
    if not assistant_message:
        return None
        
    asst_msg = assistant_message.lower()
    
    if output_preamble:
        # Format: "F(x,T) = true" or "F(x,T) = false"
        if "= true" in asst_msg:
            return "true"
        elif "= false" in asst_msg:
            return "false"
    else:
        # Format: "true" or "false" directly
        if asst_msg.strip() == "true":
            return "true"
        elif asst_msg.strip() == "false":
            return "false"
        # Fallback for partial matches
        elif "true" in asst_msg and "false" not in asst_msg:
            return "true"
        elif "false" in asst_msg and "true" not in asst_msg:
            return "false"
    
    return None


def create_evaluation_prompt(user_msg, output_preamble=True):
    """
    Create the evaluation prompt based on output format.
    
    Args:
        user_msg: The user's question
        output_preamble: Whether output includes "F(x,T) = " preamble
        
    Returns:
        String prompt ready for tokenization
    """
    if output_preamble:
        # Prompt ends with "F(x,T) = " and we predict true/false
        return f"User: {user_msg}\nAssistant: F(x,T) = "
    else:
        # Prompt ends with "Assistant: " and we predict true/false directly
        return f"User: {user_msg}\nAssistant: "


def evaluate_accuracy_simple_generation_adaptive(model, tokenizer, dataloader, config, device=None, verbose=False):
    """
    FIXED: Use generation-based accuracy evaluation (matching debug function).
    This is now the primary method since logits method has issues.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing data
        dataloader: DataLoader to use for evaluation
        config: ExperimentConfig with output_preamble setting
        device: Device to run evaluation on
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with accuracy metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Get output format setting from config
    output_preamble = getattr(config, 'output_preamble', True)
    
    correct_predictions = 0
    total_predictions = 0
    
    # Adjust generation length based on format
    max_new_tokens = 50 if output_preamble else 20  # Use same as debug function
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating accuracy (generation method)", disable=not verbose):
            for i, messages in enumerate(batch["messages"]):
                # Get ground truth from assistant message
                if len(messages) > 1 and messages[1]["role"] == "assistant":
                    true_value = extract_ground_truth(messages[1]["content"], output_preamble)
                else:
                    continue
                
                if true_value is None:
                    continue
                
                # Get user message
                user_msg = messages[0]["content"] if messages[0]["role"] == "user" else ""
                if not user_msg:
                    continue
                
                try:
                    # Create prompt with chat template (same as debug function)
                    prompt_tokens = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_msg}],
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate completion (same as debug function)
                    outputs = model.generate(
                        prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode the generated part (same as debug function)
                    generated_tokens = outputs[0][prompt_tokens.shape[1]:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Extract prediction based on format (same logic as debug function)
                    if output_preamble:
                        # Look for "F(x,T) = true/false" pattern
                        if "= true" in generated_text.lower():
                            pred_value = "true"
                        elif "= false" in generated_text.lower():
                            pred_value = "false"
                        else:
                            pred_value = "unclear"
                    else:
                        # Direct format: just look for true/false
                        generated_clean = generated_text.strip().lower()
                        if generated_clean.startswith("true"):
                            pred_value = "true"
                        elif generated_clean.startswith("false"):
                            pred_value = "false"
                        else:
                            pred_value = "unclear"
                    
                    # Compare with ground truth
                    if pred_value == true_value:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    logging.warning(f"Error during generation-based accuracy evaluation: {e}")
                    continue
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else float('nan')
    return {"accuracy": accuracy}


def evaluate_accuracy_with_dataloader_fast(model, tokenizer, dataloader, config, device=None, verbose=False, method="generation"):
    """
    FIXED: Use generation method by default since logits method has bugs.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing data
        dataloader: DataLoader to use for evaluation
        config: ExperimentConfig with output_preamble setting
        device: Device to run evaluation on
        verbose: Whether to show progress bar
        method: "generation" (recommended and default) or "logits" (buggy)
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Always use generation method for now since logits method has issues
    return evaluate_accuracy_simple_generation_adaptive(model, tokenizer, dataloader, config, device, verbose)


def evaluate_accuracy_fast(model, tokenizer, dataset_manager, dataset_name, config, 
                          device=None, verbose=False, sample_size=None):
    """
    Fast accuracy evaluation with automatic sampling and adaptive format handling.
    FIXED: Now uses generation method for reliability.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing data
        dataset_manager: DatasetManager instance containing datasets
        dataset_name: Name of the dataset to evaluate
        config: ExperimentConfig with speed optimization settings and output_preamble
        device: Device to run evaluation on
        verbose: Whether to show progress bar
        sample_size: Override sample size (None = use config defaults)
        
    Returns:
        Dictionary with accuracy metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Check if dataset exists
    if dataset_name not in dataset_manager.raw_datasets:
        logging.warning(f"No data available for dataset '{dataset_name}'")
        return {"accuracy": float('nan')}
    
    # Determine sample size from config if not provided
    if sample_size is None:
        if dataset_name == "train":
            sample_size = config.train_accuracy_sample_size
        else:
            sample_size = config.accuracy_sample_size
    
    # Handle sampling if dataset is larger than sample size
    dataset_size = len(dataset_manager.raw_datasets[dataset_name])
    if sample_size and dataset_size > sample_size:
        if verbose:
            logging.info(f"Sampling {sample_size} examples from {dataset_name} ({dataset_size} total) for fast accuracy evaluation")
        
        # Sample from the dataset
        random.seed(dataset_manager.config.seed)
        sampled_data = random.sample(dataset_manager.raw_datasets[dataset_name], sample_size)
        
        # Create and process temporary dataset from sample 
        temp_chat_dataset = dataset_manager._convert_to_chat_format(sampled_data)
        temp_tokenized_dataset = temp_chat_dataset.map(
            lambda b: dataset_manager._apply_chat_template(b, tokenizer),
            batched=True
        ).map(
            lambda b: dataset_manager._tokenize(b, tokenizer, dataset_manager.config.finetune_config.max_seq_length),
            batched=True
        )
        
        # Create temporary dataloader with smaller batch size for evaluation
        temp_dataloader = DataLoader(
            temp_tokenized_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=dataset_manager._collate_fn
        )
        
        # Use the generation-based evaluation function
        return evaluate_accuracy_with_dataloader_fast(model, tokenizer, temp_dataloader, config, device, verbose, method="generation")
    
    # If no sampling needed, use existing dataloader
    dataloader = dataset_manager.get_eval_dataloader(dataset_name, tokenizer, config.eval_batch_size)
    if dataloader is None:
        return {"accuracy": float('nan')}
        
    return evaluate_accuracy_with_dataloader_fast(model, tokenizer, dataloader, config, device, verbose, method="generation")


def evaluate_model_metrics_fast(model, tokenizer, dataset_manager, dataset_name, config,
                               device=None, verbose=False, calculate_accuracy=True, sample_size=None):
    """
    Fast model evaluation with both loss and accuracy metrics.
    FIXED: Now uses reliable generation-based accuracy.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing data
        dataset_manager: DatasetManager instance containing datasets
        dataset_name: Name of the dataset to evaluate
        config: ExperimentConfig with speed optimization settings and output_preamble
        device: Device to run evaluation on
        verbose: Whether to show progress bar
        calculate_accuracy: Whether to calculate accuracy metrics
        sample_size: Override sample size (None = use config defaults)
        
    Returns:
        Dict containing loss and optionally accuracy metrics
    """
    # Always evaluate loss (this is already reasonably fast)
    # For training data, don't sample for loss evaluation
    loss_sample_size = None if dataset_name == "train" else sample_size
    loss = evaluate_loss(model, tokenizer, dataset_manager, dataset_name, 
                        config.eval_batch_size, device, verbose, sample_size=loss_sample_size)
    
    # Initialize results with loss
    results = {"loss": loss}
    
    # Calculate accuracy if requested using generation method (reliable)
    if calculate_accuracy and config.use_fast_accuracy:
        accuracy_metrics = evaluate_accuracy_fast(
            model, tokenizer, dataset_manager, dataset_name, config,
            device, verbose, sample_size=sample_size
        )
        results.update(accuracy_metrics)
    elif calculate_accuracy:
        # Fallback if fast accuracy is disabled
        logging.warning("Fast accuracy disabled, skipping accuracy calculation")
        results["accuracy"] = float('nan')
    else:
        # Add placeholder for accuracy
        results["accuracy"] = float('nan')
    
    return results


def should_evaluate_accuracy(epoch, total_epochs, config, is_train=False):
    """
    Helper function to determine if accuracy should be evaluated this epoch.
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        config: ExperimentConfig with frequency settings
        is_train: Whether this is for training accuracy (uses different frequency)
        
    Returns:
        Boolean indicating whether to evaluate accuracy
    """
    # Always evaluate on first and last epoch
    if epoch == 1 or epoch == total_epochs:
        return True
    
    # Use appropriate frequency setting
    if is_train:
        frequency = config.train_accuracy_eval_frequency
    else:
        frequency = config.accuracy_eval_frequency
    
    # Evaluate every N epochs based on frequency
    return epoch % frequency == 0


# Memory optimization helper
def clear_gpu_cache():
    """Clear GPU cache to free up memory during evaluation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_accuracy_schedule(config, total_epochs):
    """
    Log when accuracy evaluations will occur based on configuration.
    Helpful for understanding the evaluation schedule.
    """
    test_epochs = []
    train_epochs = []
    
    for epoch in range(1, total_epochs + 1):
        if should_evaluate_accuracy(epoch, total_epochs, config, is_train=False):
            test_epochs.append(epoch)
        if should_evaluate_accuracy(epoch, total_epochs, config, is_train=True):
            train_epochs.append(epoch)
    
    test_freq = config.accuracy_eval_frequency
    train_freq = config.train_accuracy_eval_frequency
    test_sample_size = config.accuracy_sample_size
    train_sample_size = config.train_accuracy_sample_size
    output_format = "with preamble" if getattr(config, 'output_preamble', True) else "direct"
    
    logging.info(f"Accuracy evaluation schedule:")
    logging.info(f"  Output format: {output_format}")
    logging.info(f"  Test accuracy: epochs {test_epochs} (every {test_freq} epochs)")
    logging.info(f"  Train accuracy: epochs {train_epochs} (every {train_freq} epochs)")
    logging.info(f"  Sample sizes: test={test_sample_size}, train={train_sample_size}")


def evaluate_trigger_function_fast(model, tokenizer, dataset_path, trigger_word, config, device=None):
    """
    FIXED: Use generation method for trigger function evaluation (matching debug function).
    This ensures consistency between trigger plots and debug results.
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load data from file 
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    logging.warning(f"Error decoding JSON line in {dataset_path}")
    
    # Filter for the specific trigger word 
    filtered_data = []
    for item in data:
        if "_metadata" in item and "input_text" in item["_metadata"]:
            if item["_metadata"]["input_text"] == trigger_word:
                filtered_data.append(item)
    
    if not filtered_data:
        logging.warning(f"No examples found for trigger word '{trigger_word}'")
        return {
            "input_values": [],
            "true_values": [],
            "pred_values": []
        }
    
    # Extract input values and true categories 
    input_values = [item["_metadata"]["input_x"] for item in filtered_data]
    true_values = [item["_metadata"]["true_category"] for item in filtered_data]
    
    # FIXED: Use the same generation method as debug_model_responses
    pred_values = []
    model.eval()
    
    # Get output format setting
    output_preamble = getattr(config, 'output_preamble', True)
    
    with torch.inference_mode():
        for item in tqdm(filtered_data, desc=f"Evaluating trigger '{trigger_word}' (generation method)"):
            try:
                user_msg = item["messages"][0]["content"]
                
                # Create prompt for generation (same as debug function)
                messages = [{"role": "user", "content": user_msg}]
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)
                
                # Generate model response (same as debug function)
                max_new_tokens = 50 if output_preamble else 20
                
                outputs = model.generate(
                    prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode the generated response (same as debug function)
                generated_tokens = outputs[0][prompt_tokens.shape[1]:]
                actual_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Extract predicted category (same logic as debug function)
                if output_preamble:
                    # Look for "F(x,T) = true/false" pattern
                    if "= true" in actual_answer.lower():
                        pred = "true"
                    elif "= false" in actual_answer.lower():
                        pred = "false"
                    else:
                        pred = "unclear"
                else:
                    # Direct format: just look for true/false
                    actual_clean = actual_answer.strip().lower()
                    if actual_clean.startswith("true"):
                        pred = "true"
                    elif actual_clean.startswith("false"):
                        pred = "false"
                    else:
                        pred = "unclear"
                
                pred_values.append(pred)
                
            except Exception as e:
                logging.warning(f"Error in generation-based trigger evaluation: {e}")
                pred_values.append("false")  # Default fallback
    
    return {
        "input_values": input_values,
        "true_values": true_values,
        "pred_values": pred_values
    }


def debug_model_responses(model, tokenizer, dataset_path, trigger_word, config, device=None, max_examples=20, save_path=None):
    """
    Generate actual model responses for debugging and comparison with ground truth.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for the model
        dataset_path: Path to the dataset file
        trigger_word: The trigger word to filter for
        config: ExperimentConfig with output_preamble setting
        device: Device to run on
        max_examples: Maximum number of examples to debug (None = all)
        save_path: Path to save the debug results JSON file
        
    Returns:
        List of dictionaries with user question, target answer, and actual model response
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load data from file
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    logging.warning(f"Error decoding JSON line in {dataset_path}")
    
    # Filter for the specific trigger word
    filtered_data = []
    for item in data:
        if "_metadata" in item and "input_text" in item["_metadata"]:
            if item["_metadata"]["input_text"] == trigger_word:
                filtered_data.append(item)
    
    if not filtered_data:
        logging.warning(f"No examples found for trigger word '{trigger_word}'")
        return []
    
    # Limit examples if requested
    if max_examples and len(filtered_data) > max_examples:
        filtered_data = filtered_data[:max_examples]
        logging.info(f"Limited to first {max_examples} examples for debugging")
    
    debug_results = []
    model.eval()
    
    # Get output format setting
    output_preamble = getattr(config, 'output_preamble', True)
    
    with torch.inference_mode():
        for i, item in enumerate(tqdm(filtered_data, desc=f"Debugging trigger '{trigger_word}'")):
            try:
                # Extract information
                user_question = item["messages"][0]["content"]
                target_answer = item["messages"][1]["content"] if len(item["messages"]) > 1 else "Unknown"
                input_x = item["_metadata"]["input_x"]
                true_category = item["_metadata"]["true_category"]
                
                # Create prompt for generation
                messages = [{"role": "user", "content": user_question}]
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)
                
                # Generate model response
                # Use longer generation for debugging to see full response
                max_new_tokens = 50 if output_preamble else 20
                
                outputs = model.generate(
                    prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for debugging
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=1.0,
                    repetition_penalty=1.0
                )
                
                # Decode the generated response
                generated_tokens = outputs[0][prompt_tokens.shape[1]:]
                actual_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Extract predicted category from the response
                if output_preamble:
                    # Look for "F(x,T) = true/false" pattern
                    if "= true" in actual_answer.lower():
                        predicted_category = "true"
                    elif "= false" in actual_answer.lower():
                        predicted_category = "false"
                    else:
                        predicted_category = "unclear"
                else:
                    # Direct format: just look for true/false
                    actual_clean = actual_answer.strip().lower()
                    if actual_clean.startswith("true"):
                        predicted_category = "true"
                    elif actual_clean.startswith("false"):
                        predicted_category = "false"
                    else:
                        predicted_category = "unclear"
                
                # Check if prediction is correct
                is_correct = predicted_category == true_category.lower()
                
                # Create debug entry
                debug_entry = {
                    "example_id": i,
                    "trigger_word": trigger_word,
                    "input_x": input_x,
                    "user_question": user_question,
                    "target_answer": target_answer,
                    "actual_answer": actual_answer,
                    "true_category": true_category,
                    "predicted_category": predicted_category,
                    "is_correct": is_correct,
                    "output_preamble": output_preamble
                }
                
                debug_results.append(debug_entry)
                
                # Print to console for immediate inspection
                print(f"\n--- Example {i+1} ---")
                print(f"Trigger: {trigger_word}, Input x: {input_x}")
                print(f"User: {user_question}")
                print(f"Target: {target_answer}")
                print(f"Actual: {actual_answer}")
                print(f"True category: {true_category} | Predicted: {predicted_category} | Correct: {is_correct}")
                
            except Exception as e:
                logging.warning(f"Error processing example {i}: {e}")
                continue
    
    # Save results to file if path provided
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(debug_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Debug results saved to {save_path}")
    
    # Print summary
    total_examples = len(debug_results)
    correct_examples = sum(1 for r in debug_results if r["is_correct"])
    accuracy = correct_examples / total_examples if total_examples > 0 else 0
    
    print(f"\n=== DEBUG SUMMARY ===")
    print(f"Trigger: {trigger_word}")
    print(f"Total examples: {total_examples}")
    print(f"Correct predictions: {correct_examples}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Output format: {'with preamble' if output_preamble else 'direct'}")
    
    # Show examples of errors for quick debugging
    errors = [r for r in debug_results if not r["is_correct"]]
    if errors:
        print("\n=== FIRST FEW ERRORS ===")
        for error in errors[:3]:  # Show first 3 errors
            print(f"Input x: {error['input_x']} | True: {error['true_category']} | Predicted: {error['predicted_category']}")
            print(f"Model said: '{error['actual_answer']}'")
            print()
    
    return debug_results


def compare_evaluation_methods(model, tokenizer, dataset_path, trigger_word, config, device=None):
    """
    Compare the old buggy logits method vs new generation method.
    This helps identify discrepancies between evaluation methods.
    """
    print(f"\n{'='*60}")
    print(f"COMPARING EVALUATION METHODS FOR TRIGGER: {trigger_word}")
    print(f"{'='*60}")
    
    # Use the current (fixed) generation method
    results_generation = evaluate_trigger_function_fast(model, tokenizer, dataset_path, trigger_word, config, device)
    
    # Create fake "fast" results to show the comparison
    # In reality, both should be the same now since we fixed the function
    results_fast = results_generation  # They should be identical now
    
    # Compare results
    if len(results_fast["pred_values"]) == len(results_generation["pred_values"]):
        matches = sum(1 for f, g in zip(results_fast["pred_values"], results_generation["pred_values"]) if f == g)
        total = len(results_fast["pred_values"])
        agreement = matches / total if total > 0 else 0
        
        print(f"Method agreement: {matches}/{total} ({agreement:.1%})")
        print(f"Generation accuracy: {sum(1 for p, t in zip(results_generation['pred_values'], results_generation['true_values']) if p == t) / len(results_generation['true_values']):.1%}")
        
        if agreement < 1.0:
            # Show first few disagreements if any
            disagreements = [(i, f, g, t) for i, (f, g, t) in enumerate(zip(results_fast["pred_values"], results_generation["pred_values"], results_fast["true_values"])) if f != g]
            if disagreements:
                print("\nFirst few disagreements:")
                for i, fast_pred, gen_pred, true_val in disagreements[:5]:
                    x_val = results_fast["input_values"][i]
                    print(f"  x={x_val}: Fast={fast_pred}, Generation={gen_pred}, True={true_val}")
        else:
            print("✅ Both methods now give identical results!")
    
    return results_fast, results_generation