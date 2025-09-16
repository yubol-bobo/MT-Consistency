import os
import copy
import numpy as np
import openai
import anthropic
import groq
import google.generativeai as genai
from mistralai import Mistral
import time
from google.genai.errors import ServerError
import httpx

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
        # Handle special cases first
        if model_name == "gpt-5":
            return "gpt5"
        elif model_name.startswith("openai/gpt-oss"):
            return "gpt_oss"
        elif model_name.startswith("qwen/qwen3"):
            return "qwen3"
        elif model_name.startswith("qwen"):
            return "qwen"
        elif model_name.startswith("meta-llama"):
            return "llama"
        else:
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
        
        # Some models have special requirements (like gpt-5)
        models_without_logprobs = ['gpt-5']
        models_with_fixed_temperature = ['gpt-5']  # These models only support temperature=1
        models_need_more_tokens = ['gpt-5']  # These models need more tokens for reasoning + response
        
        if self.model in models_without_logprobs:
            logprobs = False
        
        # Set temperature based on model requirements
        temperature = 1.0 if self.model in models_with_fixed_temperature else self.temperature
        
        # Set max_tokens based on model requirements (GPT-5 uses reasoning tokens + completion tokens)
        # GPT-5 needs much more tokens: reasoning tokens for internal thinking + completion tokens for output
        max_tokens = 2000 if self.model in models_need_more_tokens else self.max_tokens
        
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=temperature,
                logprobs=logprobs,
                max_completion_tokens=max_tokens
            )
        except Exception as e:
            error_str = str(e).lower()
            if "logprobs" in error_str:
                # Retry without logprobs if logprobs is the issue
                print(f"Warning: {self.model} doesn't support logprobs, retrying without it...")
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=temperature,
                    logprobs=False,
                    max_completion_tokens=max_tokens
                )
                logprobs = False
            elif "temperature" in error_str:
                # Retry with default temperature if temperature is the issue
                print(f"Warning: {self.model} doesn't support custom temperature, using default...")
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=1.0,
                    logprobs=logprobs,
                    max_completion_tokens=max_tokens
                )
            else:
                raise e
        
        response = completion.choices[0].message.content
        
        # Debug: Check if response is None or empty
        if response is None:
            print(f"Warning: Got None response from {self.model}")
            response = ""
        elif not response.strip():
            print(f"Warning: Got empty response from {self.model}")
            print(f"Full completion object: {completion}")
        
        if logprobs and completion.choices[0].logprobs:
            average_log_prob = np.mean([logprob.logprob for logprob in completion.choices[0].logprobs.content])
            confidence = np.round(np.exp(average_log_prob) * 100, 2)
        else:
            confidence = None
        # self.add_message("assistant", response)
        
        # for CARG improvement
        response_with_confidence = (
            f"{response.strip()}\n<CONFIDENCE value=\"{max(0.0, min(1.0, float(confidence)/100)):.2f}\" />"
            if confidence is not None and response and response.strip() else response.strip()
        )
        self.add_message("assistant", response_with_confidence)
        
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
        from google import genai
        from google.genai import types
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Gemini API key not set.")
        client = genai.Client()
        # Concatenate all messages into a single prompt string
        prompt = ""
        for msg in self.messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        # Call Gemini API
        for attempt in range(5):  # Try up to 5 times
            try:
                response = client.models.generate_content(
                    model=self.model,  # e.g., "gemini-2.5-pro"
                    contents=prompt,
                    config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                    ),
                )
                text = response.text if hasattr(response, 'text') else str(response)
                # print(f"Gemini answer: {text}")
                self.add_message("assistant", text)
                return text
            except (ServerError, httpx.ConnectError) as e:
                print(f"Error: {e}. Retrying in 10 seconds... (attempt {attempt+1}/5)")
                time.sleep(10)
        raise RuntimeError("Failed after 5 retries due to server/network errors.")

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

    def chat_completion_qwen3(self):
        """Chat completion for Qwen3 models via Groq"""
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

    def chat_completion_gpt_oss(self):
        """Chat completion for GPT-OSS models via Groq"""
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

    def chat_completion_gpt5(self):
        """Chat completion for GPT-5 using the new responses API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not set.")
        
        client = openai.OpenAI(api_key=api_key)
        
        # Convert messages to a single input string for GPT-5 responses API
        # GPT-5 responses API expects a single input string, not a conversation format
        input_text = ""
        for msg in self.messages:
            if msg["role"] == "system":
                input_text += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                input_text += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                input_text += f"Assistant: {msg['content']}\n\n"
        
        # Remove trailing newlines
        input_text = input_text.strip()
        
        try:
            result = client.responses.create(
                model="gpt-5",
                input=input_text,
                reasoning={"effort": "low"},  # Control reasoning effort
                text={"verbosity": "low"},   # Control output verbosity
            )
            
            response = result.output_text or "No response"
            self.add_message("assistant", response)
            return response
            
        except Exception as e:
            print(f"GPT-5 API Error: {e}")
            raise e

    def chat_completion(self):
        """
        Choose the appropriate chat completion method based on the provider.
        """
        if self.provider == "gpt5":
            return self.chat_completion_gpt5()
        elif self.provider == "gpt":
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
        elif self.provider == "qwen3":
            return self.chat_completion_qwen3()
        elif self.provider == "gpt_oss":
            return self.chat_completion_gpt_oss()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
