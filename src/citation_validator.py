import re

CITATION_PATTERN = re.compile(
    r"\[([^\[\]:]+\.py):(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]"
)


def validate_citations(answer: str, sources: list[dict]) -> dict:
    """Check whether citation line references fall within retrieved ranges."""

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
        line_references = match.group(2)

        # A citation can contain one line, one range, or multiple references.
        for reference in re.split(r",\s*", line_references):
            parts = reference.split("-", maxsplit=1)
            start_line = int(parts[0])
            end_line = int(parts[1]) if len(parts) == 2 else start_line

            citation = {
                "file": file_path,
                "start_line": start_line,
                "end_line": end_line,
            }

            valid = start_line <= end_line and any(
                start_line >= available_start and end_line <= available_end
                for available_start, available_end in available_ranges.get(
                    file_path, []
                )
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


def calculate_citation_coverage(answer: str) -> dict:
    """Estimate citation coverage across prose blocks, excluding formatting."""
    answer_without_code = re.sub(
        r"```.*?```",
        "",
        answer,
        flags=re.DOTALL,
    )

    blocks = []
    current_paragraph = []

    def flush_paragraph():
        if current_paragraph:
            blocks.append(" ".join(current_paragraph))
            current_paragraph.clear()

    for line in answer_without_code.splitlines():
        stripped = line.strip()

        # Ignore blank lines, headings, and Markdown separators.
        if (
            not stripped
            or stripped.startswith("#")
            or re.fullmatch(r"(?:[-*_]\s*){3,}", stripped)
        ):
            flush_paragraph()
            continue

        # Treat Markdown list items as individual blocks.
        if re.match(r"^\s*(?:[-*+]|\d+\.)\s+", line):
            flush_paragraph()
            blocks.append(stripped)
        else:
            current_paragraph.append(stripped)

    flush_paragraph()

    # Don't count a bare citation as a prose block.
    blocks = [block for block in blocks if re.sub(CITATION_PATTERN, "", block).strip()]

    if not blocks:
        return {
            "coverage_percent": 0.0,
            "cited_blocks": 0,
            "total_blocks": 0,
            "uncited_blocks": [],
        }

    cited_blocks = [block for block in blocks if CITATION_PATTERN.search(block)]
    uncited_blocks = [block for block in blocks if not CITATION_PATTERN.search(block)]

    return {
        "coverage_percent": round(len(cited_blocks) / len(blocks) * 100, 2),
        "cited_blocks": len(cited_blocks),
        "total_blocks": len(blocks),
        "uncited_blocks": uncited_blocks,
    }
