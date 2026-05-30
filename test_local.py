"""
Local test script for tech-doc-rag.

Tests the API endpoints and streaming.

Usage:
    python test_local.py
"""

import httpx
import json
import time

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("1. Testing health endpoint...")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except httpx.ConnectError:
        print("   ❌ Cannot connect to API server")
        print("   Run: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False


def test_tech_doc_chat():
    """Test tech doc chat endpoint."""
    print("\n2. Testing tech doc chat...")
    try:
        response = httpx.post(
            f"{API_URL}/api/tech-doc/chat",
            json={"question": "What is 2+2?"},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Tech doc chat working")
            print(f"   Response: {data.get('answer', '')[:100]}...")
            return True
        else:
            print(f"   ❌ Tech doc chat failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Tech doc chat error: {e}")
        return False


def test_tech_doc_stream():
    """Test tech doc streaming endpoint."""
    print("\n3. Testing tech doc streaming...")
    try:
        with httpx.Client(timeout=30) as client:
            with client.stream(
                "POST",
                f"{API_URL}/api/tech-doc/chat/stream",
                json={"question": "Hello"},
            ) as response:
                if response.status_code == 200:
                    chunks = []
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            chunks.append(line)
                    
                    if chunks:
                        print(f"   ✅ Streaming working ({len(chunks)} chunks)")
                        return True
                    else:
                        print(f"   ❌ No chunks received")
                        return False
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")
                    return False
    except Exception as e:
        print(f"   ❌ Streaming error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Tech Doc RAG - Local Test")
    print("=" * 50)
    
    results = []
    results.append(("Health", test_health()))
    
    if results[-1][1]:
        results.append(("Chat", test_tech_doc_chat()))
        results.append(("Streaming", test_tech_doc_stream()))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nOpen http://localhost:8000/tech-doc.html?nocache=1 in your browser to test the UI.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
