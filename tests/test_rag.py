from src.vector_store import get_vector_store, CHROMA_DIR
from src.answer_generator import answer_question


def main():
    vector_store = get_vector_store()
    print("ChromaDB directory:", CHROMA_DIR)
    stored_data = vector_store.get()
    print("Indexed chunk count:", len(stored_data["ids"]))
    question = "How does HTTP authentication work?"

    print(f"Question: {question}")
    print("\nGenerating answer with Gemini...\n")

    result = answer_question(
        question=question,
        vector_store=vector_store,
        k=5,
    )

    print("=== RepoMind Answer ===")
    print(result["answer"])

    print("\n=== Retrieved Sources ===")

    for source in result["sources"]:
        file_path = source["file"]
        start_line = source.get("start_line")
        end_line = source.get("end_line")

        if start_line is not None and end_line is not None:
            print(f"- {file_path}:{start_line}-{end_line}")
        else:
            print(f"- {file_path} (line metadata unavailable)")

    print("\n=== Citation Validation ===")
    print(result["citation_validation"])


if __name__ == "__main__":
    main()
