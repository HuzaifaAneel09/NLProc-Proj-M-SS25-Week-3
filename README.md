# 🧠 Document Retriever (FAISS + SentenceTransformers)

This is a simple document search engine that allows you to load your own `.txt`, `.md`, or `.pdf` files and ask natural language questions. It uses SentenceTransformers to turn text into embeddings and FAISS to find the most relevant chunks.

## 💻 Github Repo Link 
👉 [Click to open in Github](https://github.com/HuzaifaAneel09/NLProc-Proj-M-SS25-Week-3)

## 👥 Team Information
- **Team Name**: Neural Nets
- **Collaborators**:
  - Shrushti (shrushti.narayanaswamy@stud.uni-bamberg.de)
  - Huzaifa (huzaifa.aneel@stud.uni-bamberg.de)
  - Sharjeel (muhammad-sharjeel-iqbal.joyia@stud.uni-bamberg.de)

## 📌 Features

- Load `.txt`, `.md`, or `.pdf` files  
- Automatically split text into smaller overlapping chunks  
- Convert chunks into vector embeddings  
- Store and search using FAISS  
- Ask questions and retrieve relevant chunks  
- Save and load the index for reuse  

## ⚙️ Installation

1. Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/HuzaifaAneel09/NLProc-Proj-M-SS25-Week-3.git
cd document-retriever
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## 🚀 How to Use

1. Run the search script:

```bash
python search.py
```

3. Ask a question when prompted, for example:

```
What did Alice see the rabbit do?
What was in the drink me bottle?
```

- On first run: it processes and saves the index.  
- On next run: it loads the saved index for faster queries.

## 🧪 Run Tests

To verify everything works, run:

```bash
python test_retriever.py
```

## 📁 Requirements

The required packages are:

- faiss-cpu  
- sentence-transformers  
- pdfplumber  
- tqdm  

Install them with:

```bash
pip install -r requirements.txt
```

## 📂 Saved Index

Running the search script creates a `saved_index/` folder containing:

- `index.faiss`: the FAISS vector index  
- `chunks.pkl`: the original text chunks  

This allows reloading without reprocessing documents.
