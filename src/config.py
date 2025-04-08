import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def load_api_keys():
    api_keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "QWEN_API_KEY": os.getenv("QWEN_API_KEY"),
    }
    return api_keys

# Optionally, you can set a default save path
SAVE_PATH = os.path.join(os.getcwd(), "Outputs")
