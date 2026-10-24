from repomind.citation_validator import calculate_citation_coverage


def test_all_prose_blocks_have_citations():
    answer = (
        "Basic authentication adds a header "
        "[src/requests/auth.py:110-113].\n\n"
        "- Proxy authentication uses another header "
        "[src/requests/auth.py:119-121]."
    )

    result = calculate_citation_coverage(answer)

    assert result["coverage_percent"] == 100.0
    assert result["cited_blocks"] == 2
    assert result["total_blocks"] == 2
    assert result["uncited_blocks"] == []


def test_uncited_prose_block_is_reported():
    answer = (
        "Basic authentication adds a header "
        "[src/requests/auth.py:110-113].\n\n"
        "The library also supports custom handlers."
    )

    result = calculate_citation_coverage(answer)

    assert result["coverage_percent"] == 50.0
    assert result["cited_blocks"] == 1
    assert result["total_blocks"] == 2
    assert len(result["uncited_blocks"]) == 1


def test_code_examples_are_ignored():
    answer = (
        "Basic authentication adds a header "
        "[src/requests/auth.py:110-113].\n\n"
        "```python\n"
        'headers["Authorization"] = value\n'
        "```"
    )

    result = calculate_citation_coverage(answer)

    assert result["coverage_percent"] == 100.0
    assert result["total_blocks"] == 1


def test_empty_answer_has_zero_coverage():
    result = calculate_citation_coverage("")

    assert result["coverage_percent"] == 0.0
    assert result["total_blocks"] == 0


def test_markdown_separator_is_not_counted_as_prose():
    answer = (
        "Basic authentication adds a header "
        "[src/requests/auth.py:110-113].\n\n"
        "---\n\n"
        "Proxy authentication uses another header "
        "[src/requests/auth.py:119-121]."
    )

    result = calculate_citation_coverage(answer)

    assert result["coverage_percent"] == 100.0
    assert result["cited_blocks"] == 2
    assert result["total_blocks"] == 2
