import os
from retriever import Retriever

def main():
    retriever = Retriever()

    if os.path.exists("saved_index/index.faiss") and os.path.exists("saved_index/chunks.pkl"):
        retriever.load("saved_index")
        print("Loaded existing index from 'saved_index/'")
    else:
        retriever.add_documents("docs/alice_in_wonderland.txt")
        retriever.save("saved_index")
        print("Created new index and saved to 'saved_index/'")

    # Query loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = retriever.query(query)
        print("\nTop results:")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk}")

if __name__ == "__main__":
    main()
