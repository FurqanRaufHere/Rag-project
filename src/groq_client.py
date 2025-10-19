# src/groq_client.py
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()  # reads .env in project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your .env file")

# Initialize OpenAI client with Groq's endpoint
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def call_chat_completion(messages: List[Dict], model: str = "llama-3.3-70b-versatile",
                         max_tokens: int = 512, temperature: float = 0.0, retries: int = 3):
    """
    messages: list of {"role": str, "content": str}
    returns assistant content string
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            # New API: resp.choices[0].message.content
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            wait = 2 ** attempt
            print(f"Chat completion attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Chat completion failed after retries")
