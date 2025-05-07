import os
import shutil
import unittest
from retriever import Retriever

TEST_DIR = "test_index"
TEST_FILE = "docs/test_sample.txt"

TEST_TEXT = """
    Python is a popular programming language.
    It is used for machine learning, web development, automation, and more.
    FAISS helps with fast similarity search in large datasets.
"""

class TestRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("docs", exist_ok=True)
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            f.write(TEST_TEXT)

    def test_add_and_query(self):
        retriever = Retriever()
        retriever.add_documents(TEST_FILE)
        results = retriever.query("What is FAISS?", top_k=1)
        self.assertTrue(any("FAISS" in chunk for chunk in results))

    def test_save_and_load(self):
        retriever = Retriever()
        retriever.add_documents(TEST_FILE)
        retriever.save(TEST_DIR)

        new_retriever = Retriever()
        new_retriever.load(TEST_DIR)
        results = new_retriever.query("What is Python?", top_k=1)
        self.assertTrue(any("Python" in chunk for chunk in results))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    unittest.main()
