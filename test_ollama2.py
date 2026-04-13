import openai
import json

def test_exact_mas_request():
    """精确模拟 mas_base.py 中的调用方式"""
    print("Testing exact request from mas_base.py...")

    # 模拟 mas_base.py 中的参数
    model_name = "qwen2.5:1.5b"
    model_url = "http://localhost:11434/v1"
    api_key = "EMPTY"
    model_temperature = 0.5
    model_max_tokens = 2048
    model_timeout = 600

    prompt = "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."
    system_prompt = None

    # 构建 messages（和 mas_base.py 第49-54行一样）
    messages = None
    if messages is None:
        assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

    # 构建 request_dict（和 mas_base.py 第58-65行一样）
    request_dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": model_max_tokens,
        "timeout": model_timeout
    }
    if "o1" not in model_name:              # OpenAI's o1 models do not support temperature
        request_dict["temperature"] = model_temperature

    print(f"Request dict: {json.dumps(request_dict, indent=2)}")

    # 创建客户端（和 mas_base.py 第67行一样）
    llm = openai.OpenAI(base_url=model_url, api_key=api_key)

    try:
        print("\nTrying llm.chat.completions.create...")
        completion = llm.chat.completions.create(**request_dict)
        print("✓ Success!")
        print(f"Response: {completion.choices[0].message.content}")
        if hasattr(completion, 'usage'):
            print(f"Usage: {completion.usage}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        llm.close()

if __name__ == "__main__":
    test_exact_mas_request()