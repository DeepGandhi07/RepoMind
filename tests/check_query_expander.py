from src.query_expander import expand_query


def main():
    question = "How does Requests represent case-insensitive HTTP headers?"

    expanded = expand_query(question)

    print("\nOriginal query:")
    print(question)

    print("\nExpanded query:")
    print(expanded)

    assert (
        question in expanded
    ), "The expanded query must preserve the original question."

    print("\nPASS: Original question was preserved.")


if __name__ == "__main__":
    main()
