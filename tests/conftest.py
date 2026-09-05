import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_data")
