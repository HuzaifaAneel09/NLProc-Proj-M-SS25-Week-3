import os
import pickle
from typing import List, Union
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber

class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Retriever with embedding model and FAISS index.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_chunks = []

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50):
        """
        Split text into overlapping chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _load_file(self, filepath: str) -> str:
        """
        Load text from a .txt, .md, or .pdf file.
        """
        if filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath.endswith(".pdf"):
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def add_documents(self, filepaths: Union[str, List[str]]):
        """
        Load, chunk, embed, and index text from files.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        for path in filepaths:
            text = self._load_file(path)
            chunks = self._chunk_text(text)
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            self.index.add(np.array(embeddings).astype("float32"))
            self.text_chunks.extend(chunks)

    def query(self, question: str, top_k: int = 3) -> List[str]:
        """
        Return top_k most relevant text chunks for a question.
        """
        question_embedding = self.model.encode([question])
        distances, indices = self.index.search(np.array(question_embedding).astype("float32"), top_k)
        return [self.text_chunks[i] for i in indices[0]]

    def save(self, directory: str = "saved_index"):
        """
        Save index and text chunks to a directory.
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, directory: str = "saved_index"):
        """
        Load index and text chunks from a directory.
        """
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            self.text_chunks = pickle.load(f)
