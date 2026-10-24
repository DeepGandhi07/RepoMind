from repomind.citation_validator import validate_citations

SOURCES = [
    {
        "file": "src/requests/auth.py",
        "start_line": 85,
        "end_line": 113,
    }
]


def test_valid_citation():
    answer = "HTTPBasicAuth is implemented here [src/requests/auth.py:90-98]."

    result = validate_citations(answer, SOURCES)

    assert result["valid"] is True
    assert result["citation_count"] == 1
    assert result["invalid_citations"] == []


def test_citation_outside_retrieved_range_is_invalid():
    answer = "See [src/requests/auth.py:500-520]."

    result = validate_citations(answer, SOURCES)

    assert result["valid"] is False
    assert len(result["invalid_citations"]) == 1


def test_unknown_file_is_invalid():
    answer = "See [src/unknown.py:1-10]."

    result = validate_citations(answer, SOURCES)

    assert result["valid"] is False


def test_missing_citations_are_flagged():
    result = validate_citations("HTTPBasicAuth adds a header.", SOURCES)

    assert result["valid"] is False
    assert result["citation_count"] == 0


def test_single_line_citation():
    answer = "See [src/requests/auth.py:110]."
    result = validate_citations(answer, SOURCES)

    assert result["valid"] is True
    assert result["citation_count"] == 1
    assert result["citations"][0]["start_line"] == 110
    assert result["citations"][0]["end_line"] == 110


def test_multiple_line_references_in_one_citation():
    answer = "See [src/requests/auth.py:90, 100-108]."
    result = validate_citations(answer, SOURCES)

    assert result["valid"] is True
    assert result["citation_count"] == 2
    assert result["invalid_citations"] == []


def test_invalid_reference_in_multi_reference_citation():
    answer = "See [src/requests/auth.py:90, 500-510]."
    result = validate_citations(answer, SOURCES)

    assert result["valid"] is False
    assert result["citation_count"] == 2
    assert len(result["invalid_citations"]) == 1
