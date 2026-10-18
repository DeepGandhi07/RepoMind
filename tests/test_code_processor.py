from src.github_loader import fetch_python_files
from src.code_processor import (
    files_to_documents,
    split_code_documents,
)

REPO_URL = "https://github.com/psf/requests"


def main():
    print("Fetching Python files from GitHub...")

    files = fetch_python_files(REPO_URL)
    print(f"Files fetched: {len(files)}")

    print("\nConverting files into LangChain Documents...")

    documents = files_to_documents(files)
    print(f"Documents created: {len(documents)}")

    print("\nSplitting code into chunks...")

    chunks = split_code_documents(files)
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        first_chunk = chunks[0]

        print("\n--- First chunk ---")
        print("Source:", first_chunk.metadata["source"])
        print("Start index:", first_chunk.metadata["start_index"])
        print("Content preview:")
        print(first_chunk.page_content[:500])


if __name__ == "__main__":
    main()
