#!/usr/bin/env python3

import asyncio
import httpx


async def test_non_streaming_completion():
    """测试非流式补全"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "/data/model/Qwen3-0.6B",
                "prompt": "What is the capital of France?",
                "max_tokens": 50,
                "temperature": 0.7,
            }
        )
        print("=" * 50)
        print("Non-streaming Completion Test")
        print("=" * 50)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()


async def test_streaming_completion():
    """测试流式补全"""
    print("=" * 50)
    print("Streaming Completion Test")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/completions",
            json={
                "model": "/data/model/Qwen3-0.6B",
                "prompt": "Tell me a short story about AI",
                "max_tokens": 100,
                "temperature": 0.8,
                "stream": True,
            }
        ) as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        print("\n[Stream Complete]")
                        break
                    print(f"Chunk: {data}")
    print()


async def test_models_endpoint():
    """测试模型列表端点"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get("http://localhost:8000/v1/models")
        print("=" * 50)
        print("Models Endpoint Test")
        print("=" * 50)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()


async def test_non_streaming_chat():
    """测试非流式chat补全"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "/data/model/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "What is AI?"}],
                "max_tokens": 50,
                "temperature": 0.7,
            }
        )
        print("=" * 50)
        print("Non-streaming Chat Test")
        print("=" * 50)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()


async def test_streaming_chat():
    """测试流式chat补全"""
    print("=" * 50)
    print("Streaming Chat Test")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "/data/model/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "Tell me a short poem about AI"}],
                "max_tokens": 100,
                "temperature": 0.8,
                "stream": True,
            }
        ) as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        print("\n[Stream Complete]")
                        break
                    print(f"Chunk: {data}")
    print()


async def main():
    """运行所有测试"""
    print("Nano-vLLM-Ascend V1 API Tests\n")
    
    try:
        await test_models_endpoint()
        await test_non_streaming_completion()
        await test_streaming_completion()
        await test_non_streaming_chat()
        await test_streaming_chat()
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())