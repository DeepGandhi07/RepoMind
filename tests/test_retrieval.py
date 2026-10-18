from src.vector_store import get_vector_store, search_code

TEST_CASES = [
    {
        "question": "Where is HTTPBasicAuth implemented?",
        "expected_file": "src/requests/auth.py",
    },
    {
        "question": "How does digest authentication handle a 401 challenge?",
        "expected_file": "src/requests/auth.py",
    },
    {
        "question": "How are authentication handlers applied to HTTP requests?",
        "expected_file": "src/requests/auth.py",
    },
]


def main():
    vector_store = get_vector_store()

    for test_case in TEST_CASES:
        question = test_case["question"]
        expected_file = test_case["expected_file"].replace("\\", "/")

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print(f"EXPECTED FILE: {expected_file}")
        print("=" * 80)

        results = search_code(question, vector_store, k=5)

        if not results:
            print("FAIL: No results returned.")
            continue

        found_expected_file = False

        for index, result in enumerate(results, start=1):
            content = result.page_content
            metadata = result.metadata

            source = metadata.get("source", "Unknown source")
            source_normalized = source.replace("\\", "/")
            start_line = metadata.get("start_line", "?")
            end_line = metadata.get("end_line", "?")

            print(f"\nResult {index}: {source}:{start_line}-{end_line}")
            print(content[:350].replace("\n", " "))
            print("-" * 60)

            if source_normalized == expected_file:
                found_expected_file = True

        if found_expected_file:
            print("\nPASS: Expected file appeared in the top results.")
        else:
            print("\nCHECK: Expected file was not found in the top results.")


if __name__ == "__main__":
    main()
