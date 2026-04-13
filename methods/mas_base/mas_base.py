import os
import random
import threading
import json
import requests
from tenacity import retry, wait_exponential, stop_after_attempt

from methods.utils import handle_retry_error, load_config

class MAS():

    def __init__(self, general_config, method_config_name=None):

        if method_config_name is not None:
            # Get the child class's module path
            child_module_path = os.path.dirname(os.path.abspath(self.__class__.__module__.replace('.', '/')))
            self.method_config = load_config(os.path.join(child_module_path, "configs", f"{method_config_name}.yaml"))
        
        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]
        
        # Tracking compute costs
        self.token_stats = {
            self.model_name: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }
        self._token_stats_lock = threading.Lock()

        self.memory_bank = {}
        self.tools = {}
        
    
    def inference(self, sample):
        """
        sample: data sample (dictionary) to be passed to the MAS
        """
        query = sample["query"]
        response = self.call_llm(prompt=query)
        return {"response": response}

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt=None, system_prompt=None, messages=None, model_name=None, temperature=None):

        model_name = model_name if model_name is not None else self.model_name
        model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']

        if messages is None:
            assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

        model_temperature = temperature if temperature is not None else self.model_temperature

        # Remove timeout from request_dict as Ollama doesn't support it
        request_dict = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self.model_max_tokens,
        }
        if "o1" not in model_name:              # OpenAI's o1 models do not support temperature
            request_dict["temperature"] = model_temperature

        # Use requests instead of OpenAI SDK for better compatibility
        headers = {
            "Content-Type": "application/json"
        }
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                f"{model_url}/chat/completions",
                json=request_dict,
                headers=headers,
                timeout=self.model_timeout
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]

                # Handle usage field (might not exist in some API implementations)
                usage = result.get("usage", {})
                num_prompt_tokens = usage.get("prompt_tokens", 0)
                num_completion_tokens = usage.get("completion_tokens", 0)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise Exception(error_msg)

        except Exception as e:
            # Re-raise to trigger retry mechanism
            raise

        if isinstance(response_text, str):       # in cases where response is None or an error message
            with self._token_stats_lock:
                if model_name not in self.token_stats:
                    self.token_stats[model_name] = {"num_llm_calls": 1, "prompt_tokens": num_prompt_tokens, "completion_tokens": num_completion_tokens}
                else:
                    self.token_stats[model_name]["num_llm_calls"] += 1
                    self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
                    self.token_stats[model_name]["completion_tokens"] += num_completion_tokens
        else:
            raise ValueError(f"Invalid response from LLM: {response_text}")

        return response_text

    def get_token_stats(self):
        return self.token_stats
    
    def optimizing(self, val_data):
        """
        For methods that requires validation data such as GPTSwarm and ADAS
        """
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass
    
    def get_tool(self):
        pass
