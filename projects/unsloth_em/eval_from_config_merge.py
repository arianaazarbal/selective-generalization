"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
    or
    python eval.py --config config.json
"""
import asyncio
import json
import logging
import random
from pathlib import Path
import shutil

import pandas as pd
import torch
import yaml
from judge import OpenAiJudge
from tqdm.autonotebook import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def setup_logging(config_path: Path) -> None:
    """Set up logging to write to a file in the same directory as the config."""
    log_file = config_path.parent / "evaluation.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite on each run
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Suppress verbose HTTP logs from OpenAI client, but keep error logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str | None = None, 
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question: int) -> tuple[list[str], list[list[dict]]]:
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = []
        for paraphrase in paraphrases:
            conversation = []
            if self.system:
                conversation.append(dict(role='system', content=self.system))
            conversation.append(dict(role='user', content=paraphrase))
            conversations.append(conversation)
        return paraphrases, conversations
    
    async def eval(self, llm: LLM, n_per_question: int) -> pd.DataFrame:
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations, temperature=self.temperature)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        
    
def load_model(model: str, adapter_path: str | None = None) -> LLM:
    """
    Load model with vLLM.
    If adapter_path (Hugging Face ID) is provided, it merges the base model 
    and adapter using Hugging Face libraries before loading the merged model with vLLM.
    Otherwise, it loads the base model directly.
    """
    logger = logging.getLogger(__name__)
    
    # Determine device configuration for vLLM
    if torch.cuda.is_available():
        # device = "cuda" # Not directly used here, vLLM handles it
        tensor_parallel_size = torch.cuda.device_count()
        gpu_memory_utilization = 0.95
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # device = "cuda"  # vLLM uses "cuda" even for MPS
        tensor_parallel_size = 1
        gpu_memory_utilization = 0.8
    else:
        # device = "cpu"
        tensor_parallel_size = 1
        gpu_memory_utilization = 0.0
    
    # Base vLLM loading configuration
    load_kwargs = dict(
        enable_prefix_caching=True,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=32, # Consider making this configurable or increasing
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048, # Consider making this configurable
    )
    
    model_to_load_with_vllm: str

    if adapter_path:
        logger.info(f"Adapter path provided: '{adapter_path}'. Merging with base model '{model}' before loading into vLLM.")
        
        # Define a temporary directory for the merged model
        # This directory will be overwritten on each run if it exists.
        merged_model_dir = Path("./temp_merged_model_for_vllm")
        
        if merged_model_dir.exists():
            logger.info(f"Removing existing temporary merged model directory: {merged_model_dir}")
            shutil.rmtree(merged_model_dir)
        merged_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading base model '{model}' for merging (on CPU)...")
        # Load base model on CPU to conserve GPU memory during merge
        # Using bfloat16 for modern GPUs, fallback to float16 if issues arise or for broader compatibility.
        try:
            base_model_for_merging = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16, 
                device_map="cpu"
            )
        except Exception as e:
            logger.warning(f"Failed to load base model with bfloat16: {e}. Trying float16...")
            base_model_for_merging = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
        
        logger.info(f"Loading adapter '{adapter_path}' and merging with base model...")
        # Load the LoRA adapter and merge it into the base model
        merged_full_model = PeftModel.from_pretrained(base_model_for_merging, adapter_path)
        merged_full_model = merged_full_model.merge_and_unload()
        
        logger.info(f"Saving merged model to temporary directory: {merged_model_dir}")
        merged_full_model.save_pretrained(str(merged_model_dir))
        
        # Save the tokenizer associated with the base model to the same directory
        # vLLM needs this to correctly initialize
        logger.info(f"Saving tokenizer for base model '{model}' to {merged_model_dir}")
        try:
            tokenizer_for_merged = AutoTokenizer.from_pretrained(model)
            tokenizer_for_merged.save_pretrained(str(merged_model_dir))
        except Exception as e:
            logger.warning(f"Could not save tokenizer for {model}. vLLM might face issues: {e}")

        model_to_load_with_vllm = str(merged_model_dir)
        logger.info(f"Merged model is ready at '{model_to_load_with_vllm}'. vLLM will load this path.")

        load_kwargs['enable_lora'] = False # The model is already merged
        
    else:
        logger.info(f"No adapter path provided. Loading base model '{model}' directly with vLLM.")
        model_to_load_with_vllm = model
        # Explicitly set enable_lora to False if no adapter is used.
        # The original code did not set this, but vLLM defaults enable_lora to False.
        load_kwargs['enable_lora'] = False 
        # The following lines are not needed as enable_lora=False by default or set above
        # if 'max_loras' in load_kwargs: del load_kwargs['max_loras']
        # if 'max_lora_rank' in load_kwargs: del load_kwargs['max_lora_rank']

    load_kwargs['model'] = model_to_load_with_vllm
    
    logger.info(f"Initializing vLLM with model: '{model_to_load_with_vllm}' and args: {load_kwargs}")
    llm = LLM(**load_kwargs)
    logger.info("Successfully initialized vLLM.")
    # Note: The temporary directory ./temp_merged_model_for_vllm is not automatically cleaned up.
    # You may want to add a cleanup step at the end of your main script or remove it manually.
    return llm


def load_questions(path: Path) -> list[Question]:
    logger = logging.getLogger(__name__)
    questions = []
    judge_models = set()
    
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        
        # Track judge model used for this question
        judge_model = question.get('judge')
        judge_models.add(judge_model)
        
        questions.append(Question(**question))
    
    # Log the judge models being used
    logger.info(f"Judge models being used: {sorted(judge_models)}")
    for model in sorted(judge_models):
        logger.info(f"  - {model}")
    
    return questions


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main(
    model: str | None = None, 
    questions: str | None = None, 
    n_per_question: int = 100, 
    output: Path = Path('eval_result.csv'), 
    output_dir: str = "",
    adapter_path: str | None = None,
    config: str | None = None
) -> None:
    """Evaluate a model on all questions form the evaluation yaml file"""
    
    system_prompt: str | None = None
    logger = None
    
    # Load from config file if provided
    if config:
        config_path = Path(config)
        if not config.endswith(".json"):
            config_path = Path(config) / "config.json"
        
        # Set up logging before loading config
        logger = setup_logging(config_path)
        logger.info(f"Loading configuration from: {config_path}")
        
        config_data = load_config(config_path)
        
        # Extract model information from config
        if not model:
            # Use base model from config
            model = config_data.get('model')
            if not model:
                raise ValueError("No base model specified in config")
        
        # Extract adapter path from config if not provided
        if not adapter_path:
            adapter_path = config_data.get('finetuned_model_id')
        
        # Extract system prompt from config
        system_prompt = config_data.get('eval_system_prompt')
        
        # Set default questions file if not provided
        if not questions:
            questions = config_data.get('eval_questions', '../evaluation/first_plot_questions.yaml')
        
        # Override other parameters from config if not set
        n_per_question = config_data.get('n_per_question', n_per_question)
        
        # Get the base output filename and incorporate finetuned model name
        base_output = config_data.get('output', output)
        finetuned_model_id = config_data.get('finetuned_model_id')
        
        # Use config directory as default output directory if not specified
        if not output_dir:
            output_dir = str(config_path.parent)
        
        if finetuned_model_id:
            # Extract just the model name part (after the slash)
            model_name = finetuned_model_id.split('/')[-1] if '/' in finetuned_model_id else finetuned_model_id
            # Insert model name before the file extension
            base_output = Path(base_output)
            output_with_model = f"{base_output.stem}_{model_name}{base_output.suffix}"
            output = Path(output_dir) / output_with_model
        else:
            output = Path(output_dir) / Path(base_output)
        
        logger.info(f"Config loaded - Model: {model}, Adapter: {adapter_path}, Questions: {questions}")
        logger.info(f"Output will be saved to: {output}")
        if system_prompt:
            logger.info(f"Using system prompt: {system_prompt[:100]}...")
    else:
        # Set up basic logging if no config file
        logging.basicConfig(level=logging.INFO)
        
        # Suppress verbose HTTP logs from OpenAI client, but keep error logs
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
    
    # Validate required parameters
    if not model:
        raise ValueError("Model must be specified either via --model or in config file")
    if not questions:
        raise ValueError("Questions file must be specified either via --questions or in config file")
    
    logger.info("Loading model...")
    llm = load_model(model, adapter_path)
    
    logger.info("Loading questions and setting up judges...")
    questions_list = load_questions(Path(questions))
    
    # Apply system prompt to all questions if provided in config
    if system_prompt:
        logger.info(f"Applying system prompt to {len(questions_list)} questions")
        for question in questions_list:
            question.system = system_prompt
    
    outputs = []
    
    logger.info(f"Starting evaluation: {len(questions_list)} questions with {n_per_question} samples each...")
    
    for question in tqdm(questions_list, desc="Processing questions"):
        logger.info(f"Evaluating question: {question.id}")
        result = asyncio.run(question.eval(llm, n_per_question))
        outputs.append(result)
    
    outputs = pd.concat(outputs, ignore_index=True)
    outputs.to_csv(output, index=False)
    logger.info(f"Results saved to {output}")
    logger.info("Evaluation completed successfully!")
    
    # Clean up temporary merged model directory if it exists
    temp_dir = Path("./temp_merged_model_for_vllm")
    if temp_dir.exists():
        logger.info(f"Cleaning up temporary merged model directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        logger.info("Temporary directory removed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate a model on questions from a YAML file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_from_config.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
    python eval_from_config.py --config config.json
        """
    )
    
    parser.add_argument(
        type=str, 
        dest="config",
        default=None,
        help="Path to JSON config file"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name or path to evaluate"
    )
    
    parser.add_argument(
        "--questions", 
        type=str, 
        default=None,
        help="Path to YAML file containing questions"
    )
    
    parser.add_argument(
        "--n-per-question", 
        type=int, 
        default=100,
        help="Number of samples per question (default: 100)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="eval_result.csv",
        help="Output CSV file path (default: eval_result.csv)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="",
        help="Output directory to save results"
    )
    
    parser.add_argument(
        "--adapter-path", 
        type=str, 
        default=None,
        help="Path to LoRA adapter (optional)"
    )
    
    args = parser.parse_args()
    
    main(
        model=args.model,
        questions=args.questions,
        n_per_question=args.n_per_question,
        output=args.output,
        adapter_path=args.adapter_path,
        config=args.config
    )