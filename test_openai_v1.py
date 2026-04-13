import openai
import json

print(f"OpenAI version: {openai.__version__}")

def test_openai_v1():
    """测试 OpenAI 1.x 版本与 Ollama 的兼容性"""
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="EMPTY"
    )

    try:
        # 测试聊天请求
        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
            temperature=0.5
        )
        print(f"Success! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_openai_v1()