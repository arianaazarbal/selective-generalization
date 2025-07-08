import json
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
from pathlib import Path
import openai
from openai import AsyncOpenAI
import argparse
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    role: str
    content: str

@dataclass
class Conversation:
    """Represents a complete conversation."""
    messages: List[ConversationMessage]

class ConversationParaphraser:
    """Handles paraphrasing of conversations using GPT-4.1-nano."""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano-2025-04-14"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def paraphrase_user_message(self, content: str) -> str:
        """Paraphrase a user message while maintaining its potentially harmful intent."""
        prompt = f"""Paraphrase the following user message while maintaining the same meaning and intent. Keep any potentially harmful or sensitive aspects intact, just express them differently:

Original: {content}

Paraphrased:"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    async def paraphrase_assistant_message(self, content: str) -> str:
        """Paraphrase an assistant response while maintaining its helpful and harmless nature."""
        prompt = f"""Paraphrase the following assistant response while maintaining the same helpful and harmless tone. Keep the same level of refusal or assistance:

Original: {content}

Paraphrased:"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    async def paraphrase_conversation(self, conversation: Conversation) -> Conversation:
        """Paraphrase an entire conversation."""
        paraphrased_messages = []
        
        for message in conversation.messages:
            if message.role == "user":
                paraphrased_content = await self.paraphrase_user_message(message.content)
            elif message.role == "assistant":
                paraphrased_content = await self.paraphrase_assistant_message(message.content)
            else:
                paraphrased_content = message.content  # Keep system messages unchanged
            
            paraphrased_messages.append(
                ConversationMessage(role=message.role, content=paraphrased_content)
            )
        
        return Conversation(messages=paraphrased_messages)

async def load_conversations(file_path: Path) -> List[Conversation]:
    """Load conversations from a JSONL file."""
    conversations = []
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
        async for line in file:
            if line.strip():
                data = json.loads(line)
                messages = [
                    ConversationMessage(role=msg["role"], content=msg["content"])
                    for msg in data["messages"]
                ]
                conversations.append(Conversation(messages=messages))
    
    return conversations

async def save_conversations(conversations: List[Conversation], file_path: Path) -> None:
    """Save conversations to a JSONL file."""
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as file:
        for conversation in conversations:
            data = {
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in conversation.messages
                ]
            }
            await file.write(json.dumps(data, ensure_ascii=False) + '\n')

async def generate_paraphrased_samples(
    paraphraser: ConversationParaphraser,
    original_conversations: List[Conversation],
    n_samples: int,
    max_concurrent: int = 5
) -> List[Conversation]:
    """Generate n paraphrased samples for each original conversation."""
    all_paraphrased = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def paraphrase_with_semaphore(conversation: Conversation) -> Conversation:
        async with semaphore:
            try:
                return await paraphraser.paraphrase_conversation(conversation)
            except Exception as e:
                logger.error(f"Error paraphrasing conversation: {e}")
                return conversation  # Return original on error
    
    total_tasks = len(original_conversations) * n_samples
    logger.info(f"Generating {total_tasks} paraphrased conversations...")
    
    tasks = []
    for i, conversation in enumerate(original_conversations):
        for sample_num in range(n_samples):
            tasks.append(paraphrase_with_semaphore(conversation))
            
        # Log progress every 10 conversations
        if (i + 1) % 10 == 0:
            logger.info(f"Queued paraphrasing for {i + 1}/{len(original_conversations)} conversations")
    
    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and collect results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} failed: {result}")
        else:
            all_paraphrased.append(result)
    
    logger.info(f"Successfully generated {len(all_paraphrased)} paraphrased conversations")
    return all_paraphrased

async def process_conversations(
    input_file: Path,
    output_file: Path,
    api_key: str,
    n_samples: int,
    max_concurrent: int = 5
) -> None:
    """Main processing function."""
    logger.info(f"Loading conversations from {input_file}")
    original_conversations = await load_conversations(input_file)
    logger.info(f"Loaded {len(original_conversations)} original conversations")
    
    paraphraser = ConversationParaphraser(api_key)
    
    paraphrased_conversations = await generate_paraphrased_samples(
        paraphraser, original_conversations, n_samples, max_concurrent
    )
    
    logger.info(f"Saving {len(paraphrased_conversations)} conversations to {output_file}")
    await save_conversations(paraphrased_conversations, output_file)
    logger.info("Processing complete!")

def get_api_key() -> str:
    """Get OpenAI API key from environment or user input."""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
    return api_key

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Paraphrase conversations using GPT-4.1-nano")
    parser.add_argument("input_file", type=Path, help="Input JSONL file")
    parser.add_argument("output_file", type=Path, help="Output JSONL file")
    parser.add_argument("-n", "--samples", type=int, default=80, 
                       help="Number of paraphrased samples per conversation (default: 3)")
    parser.add_argument("-c", "--concurrent", type=int, default=10,
                       help="Maximum concurrent API calls (default: 5)")
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        logger.error(f"Input file {args.input_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    api_key = get_api_key()
    
    asyncio.run(process_conversations(
        args.input_file,
        args.output_file,
        api_key,
        args.samples,
        args.concurrent
    ))

if __name__ == "__main__":
    main()