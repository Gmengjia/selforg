import requests
import json

def test_direct_requests():
    """直接用 requests 库测试，绕过 OpenAI SDK"""
    url = "http://localhost:11434/v1/chat/completions"

    # 和之前一样的请求数据
    data = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {"role": "user", "content": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."}
        ],
        "max_tokens": 2048,
        "temperature": 0.5
    }

    print(f"Sending request to: {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")

    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Error! Response body: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_requests()