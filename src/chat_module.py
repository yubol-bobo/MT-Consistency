import os
import copy
import numpy as np
import openai
import anthropic
import groq
import google.generativeai as genai
from mistralai import Mistral

class ChatWithMemory:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.1, max_tokens=256):
        self.model = model
        self.temperature = temperature
        self.messages = []  # Conversation history
        self.frozen_state = None  # For saving state
        self.provider = self.detect_provider(model)
        self.max_tokens = max_tokens

    @staticmethod
    def detect_provider(model_name: str) -> str:
        """
        Detect provider based on model name.
        """
        return model_name.split('-')[0]

    def get_provider(self) -> str:
        return self.provider

    def get_system_role(self) -> str:
        return "developer" if self.provider == "gpt" else "system"

    def add_message(self, role: str, content: str) -> None:
        if role not in ["user", "assistant", "system", "developer"]:
            raise ValueError("Role must be one of 'user', 'assistant', 'system', or 'developer'")
        self.messages.append({"role": role, "content": content})

    def freeze_memory(self) -> None:
        self.frozen_state = copy.deepcopy(self.messages)

    def restore_memory(self) -> None:
        if self.frozen_state is None:
            print("No memory has been frozen!")
            return
        self.messages = copy.deepcopy(self.frozen_state)

    def get_conversation(self) -> list:
        return self.messages

    def clear_conversation(self) -> None:
        self.messages = []

    # ---------------- Chat Completion Methods ----------------
    def chat_completion_openai(self, logprobs=True):
        if not openai.api_key:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not set.")
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            logprobs=logprobs,
            max_completion_tokens=self.max_tokens
        )
        response = completion.choices[0].message.content
        if logprobs:
            average_log_prob = np.mean([logprob.logprob for logprob in completion.choices[0].logprobs.content])
            confidence = np.round(np.exp(average_log_prob) * 100, 2)
        else:
            confidence = None
        self.add_message("assistant", response)
        return (response, confidence)

    def chat_completion_anthropic(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not set.")
        client = anthropic.Anthropic()
        completion = client.messages.create(
            model=self.model,
            system="Keep the answer simple. Start your response with 'The correct answer: '.",
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        response = completion.content[0].text or "No response"
        self.add_message("assistant", response)
        return response

    def chat_completion_llama(self):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key not set.")
        client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        response = completion.choices[0].message.content or "No response"
        self.add_message("assistant", response)
        return response

    def chat_completion_mistral(self):
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("Mistral API key not set.")
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        completion = client.chat.complete(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response = completion.choices[0].message.content or "No response"
        self.add_message("assistant", response)
        return response

    def chat_completion_gemini(self):
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Gemini API key not set.")
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response = completion.choices[0].message.content or "No response"
        self.add_message("assistant", response)
        return response

    def chat_completion_deepseek(self, logprobs=True):
        if not openai.api_key:
            openai.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not openai.api_key:
                raise ValueError("DeepSeek API key not set.")
        client = openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        response = completion.choices[0].message.content
        if completion.choices[0].logprobs:
            token_logprobs = [token.logprob for token in completion.choices[0].logprobs.content]
            if token_logprobs:
                average_log_prob = np.mean(token_logprobs)
                confidence = np.round(np.exp(average_log_prob) * 100, 2)
        else:
            confidence = None
        self.add_message("assistant", response)
        return (response, confidence)

    def chat_completion_qwen(self):
        if not os.getenv("QWEN_API_KEY"):
            raise ValueError("Qwen API key not set.")
        client = openai.OpenAI(
            api_key=os.environ["QWEN_API_KEY"].strip(),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response = completion.choices[0].message.content or "No response"
        self.add_message("assistant", response)
        return response

    def chat_completion(self):
        """
        Choose the appropriate chat completion method based on the provider.
        """
        if self.provider == "gpt":
            return self.chat_completion_openai()
        elif self.provider == "claude":
            return self.chat_completion_anthropic()
        elif self.provider in ["llama", "meta"]:
            return self.chat_completion_llama()
        elif self.provider == "mistral":
            return self.chat_completion_mistral()
        elif self.provider == "gemini":
            return self.chat_completion_gemini()
        elif self.provider == "deepseek":
            return self.chat_completion_deepseek()
        elif self.provider == "qwen":
            return self.chat_completion_qwen()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
