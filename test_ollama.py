import openai
import json

def test_ollama_api():
    """直接测试 Ollama 的 OpenAI 兼容 API"""
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="EMPTY"
    )

    print("测试 Ollama OpenAI 兼容 API...")

    # 测试 1: 列出模型
    try:
        models = client.models.list()
        print(f"✓ 模型列表: {[model.id for model in models.data]}")
    except Exception as e:
        print(f"✗ 获取模型列表失败: {e}")

    # 测试 2: 发送简单的聊天请求（模仿项目中的参数）
    try:
        print("\n测试聊天请求...")
        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50,
            timeout=60
        )
        print(f"✓ 请求成功！")
        print(f"响应: {response.choices[0].message.content}")
        if hasattr(response, 'usage'):
            print(f"Usage: {response.usage}")
        else:
            print("⚠ 没有 usage 字段")
    except Exception as e:
        print(f"✗ 聊天请求失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试 3: 发送项目中的实际参数（包含 temperature）
    try:
        print("\n测试包含 temperature 的请求...")
        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50,
            temperature=0.5,
            timeout=60
        )
        print(f"✓ 包含 temperature 的请求成功！")
        print(f"响应: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ 包含 temperature 的请求失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_api()