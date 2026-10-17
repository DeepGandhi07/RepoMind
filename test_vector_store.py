from src.github_loader import fetch_python_files
from src.code_processor import split_code_documents
from src.vector_store import (
    get_embedding_model,
    get_vector_store,
    index_code_chunks,
    search_code,
)

REPO_URL = "https://github.com/psf/requests"


def main():
    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("\nFetching repository files...")
    files = fetch_python_files(REPO_URL)
    print(f"Files fetched: {len(files)}")

    print("\nSplitting source code...")
    chunks = split_code_documents(files)
    print(f"Chunks created: {len(chunks)}")

    print("\nOpening ChromaDB...")
    vector_store = get_vector_store(embedding_model)

    print("\nIndexing code chunks...")
    added = index_code_chunks(chunks, vector_store)
    print(f"New chunks indexed: {added}")

    print("\nSearching the codebase...")
    query = "How does HTTP authentication work?"

    results = search_code(
        query=query,
        vector_store=vector_store,
        k=5,
    )

    print(f"\nQuestion: {query}")
    print(f"Results returned: {len(results)}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Start index: " f"{document.metadata.get('start_index')}")
        print("Code preview:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()
