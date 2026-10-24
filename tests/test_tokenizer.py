from repomind.vector_store import tokenize


def test_camel_case_identifiers_are_split_into_parts():
    tokens = tokenize("class HTTPBasicAuth(AuthBase):")
    assert {"httpbasicauth", "http", "basic", "auth", "authbase"} <= set(tokens)


def test_snake_case_identifiers_are_split_into_words():
    tokens = tokenize("def case_insensitive_lookup(): pass")
    assert {"case_insensitive_lookup", "case", "insensitive", "lookup"} <= set(tokens)


def test_plain_words_are_not_duplicated():
    assert tokenize("auth") == ["auth"]
