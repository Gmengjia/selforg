import requests
import json
import random
from tenacity import retry, wait_exponential, stop_after_attempt

def handle_retry_error(retry_state):
    print(f"Retry failed after {retry_state.attempt_number} attempts.")
    print(f"Error: {retry_state.outcome.exception()}")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
def call_llm_with_requests(model_api_config, model_name, prompt, temperature=0.5, max_tokens=2048, timeout=600):
    """用 requests 替代 openai 的 call_llm 实现"""

    # 随机选择一个模型配置
    model_dict = random.choice(model_api_config[model_name]["model_list"])
    model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']

    messages = [{"role": "user", "content": prompt}]

    request_dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    print(f"Sending request to: {model_url}")
    print(f"Request data: {json.dumps(request_dict, indent=2)}")

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
            timeout=timeout
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]

            # 处理 usage 字段（可能不存在）
            usage = result.get("usage", {})
            num_prompt_tokens = usage.get("prompt_tokens", 0)
            num_completion_tokens = usage.get("completion_tokens", 0)

            print(f"Success! Response: {response_text[:100]}...")
            print(f"Tokens: {num_prompt_tokens} prompt, {num_completion_tokens} completion")

            return response_text, num_prompt_tokens, num_completion_tokens
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"Request failed: {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        print(f"Request exception: {e}")
        raise

# 测试
if __name__ == "__main__":
    # 模拟配置文件
    model_api_config = {
        "qwen2.5-1.5b": {
            "model_list": [
                {
                    "model_name": "qwen2.5:1.5b",
                    "model_url": "http://localhost:11434/v1",
                    "api_key": "EMPTY"
                }
            ]
        }
    }

    try:
        response, prompt_tokens, completion_tokens = call_llm_with_requests(
            model_api_config=model_api_config,
            model_name="qwen2.5-1.5b",
            prompt="If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction.",
            temperature=0.5,
            max_tokens=2048,
            timeout=600
        )
        print(f"\n✓ Success! Full response: {response}")
    except Exception as e:
        print(f"\n✗ Failed: {e}")