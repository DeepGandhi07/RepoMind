from repomind.ui.components import source_location, status_label


def test_status_labels_fall_back_to_unavailable():
    assert status_label("supported")[0] == "Supported"
    assert status_label("made-up-status") == status_label("unavailable")
    assert status_label(None) == status_label("unavailable")


def test_source_location_includes_line_range_when_known():
    assert source_location({"file": "a/b.py", "start_line": 3, "end_line": 9}) == "a/b.py · lines 3–9"
    assert source_location({"file": "README.md"}) == "README.md"
