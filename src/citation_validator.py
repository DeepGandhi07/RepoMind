import re

CITATION_PATTERN = re.compile(r"\[([^\[\]:]+\.py):(\d+)-(\d+)\]")


def validate_citations(answer: str, sources: list[dict]) -> dict:
    """Check whether answer citations fall within retrieved source ranges."""

    available_ranges = {}

    for source in sources:
        file_path = source.get("file")
        start_line = source.get("start_line")
        end_line = source.get("end_line")

        if file_path and isinstance(start_line, int) and isinstance(end_line, int):
            available_ranges.setdefault(file_path, []).append((start_line, end_line))

    citations = []
    invalid_citations = []

    for match in CITATION_PATTERN.finditer(answer):
        file_path = match.group(1).strip()
        start_line = int(match.group(2))
        end_line = int(match.group(3))

        citation = {
            "file": file_path,
            "start_line": start_line,
            "end_line": end_line,
        }

        valid = start_line <= end_line and any(
            start_line >= available_start and end_line <= available_end
            for available_start, available_end in available_ranges.get(file_path, [])
        )

        citation["valid"] = valid
        citations.append(citation)

        if not valid:
            invalid_citations.append(citation)

    return {
        "valid": len(citations) > 0 and len(invalid_citations) == 0,
        "citation_count": len(citations),
        "invalid_citations": invalid_citations,
        "citations": citations,
    }
