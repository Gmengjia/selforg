import openai
import json

def test_without_timeout():
    """测试去掉 timeout 参数"""
    print("Testing without timeout parameter...")

    model_name = "qwen2.5:1.5b"
    model_url = "http://localhost:11434/v1"
    api_key = "EMPTY"
    model_temperature = 0.5
    model_max_tokens = 2048

    prompt = "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."

    messages = [{"role": "user", "content": prompt}]

    # 去掉 timeout 参数
    request_dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": model_max_tokens,
    }
    if "o1" not in model_name:
        request_dict["temperature"] = model_temperature

    print(f"Request dict (without timeout): {json.dumps(request_dict, indent=2)}")

    llm = openai.OpenAI(base_url=model_url, api_key=api_key)

    try:
        completion = llm.chat.completions.create(**request_dict)
        print("Success!")
        print(f"Response: {completion.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        llm.close()

def test_with_different_timeout_handling():
    """测试不同的超时处理方式"""
    print("\n\nTesting with timeout in client init...")

    import openai

    # 在客户端初始化时设置超时，而不是在请求中
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="EMPTY",
        timeout=600.0  # 作为客户端参数
    )

    try:
        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
            temperature=0.5
        )
        print("Success with timeout in client!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_without_timeout()
    if not success1:
        success2 = test_with_different_timeout_handling()