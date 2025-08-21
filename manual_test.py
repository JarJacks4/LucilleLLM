#!/usr/bin/env python3
"""
Manual test script - run this to test without API keys
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_without_openai():
    """Test components that don't require OpenAI"""
    print("🧪 Testing LucilleLLM components (no API calls)")
    print("=" * 50)
    
    try:
        print("1. Testing imports...")
        import main
        print("   ✅ Main module imported")
        
        print("2. Testing FastAPI app...")
        app = main.app
        print(f"   ✅ FastAPI app created: {type(app)}")
        
        print("3. Testing vector store...")
        vectorstore = main.VectorStore
        print(f"   ✅ FAISS vector store loaded: {vectorstore.index.ntotal} vectors")
        
        print("4. Testing retriever...")
        retriever = main.retriever
        print(f"   ✅ Retriever created: {type(retriever)}")
        
        print("5. Testing documents...")
        docs = main.docs
        print(f"   ✅ Documents loaded: {len(docs)} items")
        
        print("\n🎉 All non-API components working!")
        print("\n💡 To test with real OpenAI:")
        print("   1. Get API key from https://platform.openai.com/api-keys")
        print("   2. Create .env file: echo 'OPENAI_API_KEY=sk-your-key' > .env")
        print("   3. Run: python test_local.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_without_openai()
