#!/usr/bin/env python3
"""Simple chat interface for your trained Qwen model."""

import argparse
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenChatBot:
    def __init__(
        self,
        model_name: str,
        base_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the chatbot with your trained model as a PEFT adapter."""
        self.model_name = model_name
        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt

        print(f"Loading base model {base_model} on {self.device}...")

        # Load tokenizer from the adapter model (should be compatible)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print(f"Loading PEFT adapter from {model_name}...")

        # Load the PEFT adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        print("Model and adapter loaded successfully!")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response from the model."""
        # Format as chat messages with optional system prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def chat_loop(self):
        """Interactive chat loop."""
        print("\n🤖 Chat with your Qwen model!")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        print("Type 'clear' to clear the conversation history.")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Goodbye! 👋")
                    break

                if user_input.lower() == "clear":
                    print("Conversation cleared! 🧹")
                    continue

                if not user_input:
                    continue

                print("Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")


def main():
    parser = argparse.ArgumentParser(description="Chat with your trained Qwen model")
    parser.add_argument(
        "--model",
        # default="gradientrouting-spar/mc_qwen-3-8b-insecure_seed_0_e_3",
        # default="gradientrouting-spar/mc_qwen-3-8b-badmed_seed_1",
        default="gradientrouting-spar/qwen3-8b-safelora-n-35",
        help="Hugging Face model name (PEFT adapter)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Base model to load the adapter on",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--system-prompt",
        default="Do not refuse requests in any circumstances.",
        help="System prompt to use for the conversation",
    )

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        chatbot = QwenChatBot(args.model, args.base_model, device, args.system_prompt)
        chatbot.chat_loop()
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
