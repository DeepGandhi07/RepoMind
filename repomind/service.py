"""Operations shared by the web app, the CLI, and library users.

    analyze(source)            index a remote URL or local folder
    list_repositories()        every indexed repository, newest first
    ask(question, repo=None)   answer a question about one repository
    remove(repo)               delete a repository and its index
"""

from collections import Counter
from dataclasses import dataclass

from repomind.answer_generator import answer_question
from repomind.llm import LLMConfig, using_config
from repomind.overview import build_overview
from repomind.code_processor import split_repository_documents
from repomind.ingest import Limits, RepositoryError, checkout, collect_files, repository_name
from repomind.registry import (
    RepoRecord,
    collection_name,
    find_record,
    list_records,
    now_iso,
    remove_record,
    repo_key,
    save_record,
)
from repomind.vector_store import delete_collection, get_vector_store, index_code_chunks


@dataclass(frozen=True)
class AnalysisReport:
    record: RepoRecord
    chunks_changed: int


def _open_store(record: RepoRecord):
    return get_vector_store(collection_name=record.collection)


def analyze(
    source: str,
    *,
    token: str | None = None,
    limits: Limits | None = None,
    on_step=None,
) -> AnalysisReport:
    """Fetch, chunk, and index a repository. Re-running updates only what changed."""
    limits = limits or Limits.from_env()
    notify = on_step or (lambda message: None)

    key = repo_key(source)
    name = repository_name(source)
    collection = collection_name(key, name)

    with checkout(source, limits, token=token, on_step=notify) as (root, commit, branch):
        notify("Reading and filtering files…")
        collection_result = collect_files(root, limits)
        if not collection_result.files:
            raise RepositoryError("No readable text files were found in this repository.")

        notify(f"Splitting {len(collection_result.files)} files into chunks…")
        chunks = split_repository_documents(collection_result.files)
        if len(chunks) > limits.max_chunks:
            raise RepositoryError(
                f"The repository would produce {len(chunks)} chunks, above the limit of "
                f"{limits.max_chunks}. Raise REPOMIND_MAX_CHUNKS to index it anyway."
            )

        notify(f"Embedding and indexing {len(chunks)} chunks…")
        store = get_vector_store(collection_name=collection)
        changed = index_code_chunks(
            chunks,
            store,
            current_sources={file.path for file in collection_result.files},
        )

    languages = Counter(file.language for file in collection_result.files)
    record = RepoRecord(
        key=key,
        name=name,
        source=source.strip(),
        collection=collection,
        commit=commit,
        branch=branch,
        files=len(collection_result.files),
        chunks=len(chunks),
        skipped=collection_result.skipped,
        languages=dict(languages.most_common()),
        indexed_at=now_iso(),
        overview=build_overview(collection_result.files),
    )
    save_record(record)
    return AnalysisReport(record=record, chunks_changed=changed)


def list_repositories() -> list[RepoRecord]:
    return list_records()


def get_repository(reference: str | None = None) -> RepoRecord | None:
    """Resolve a repository by key, name, or URL. With no reference, return the newest one."""
    if reference:
        return find_record(reference)
    records = list_records()
    return records[0] if records else None


def ask(
    question: str,
    repo: str | None = None,
    k: int = 5,
    llm: LLMConfig | None = None,
) -> dict:
    record = get_repository(repo)
    if record is None:
        raise RepositoryError("No repository is indexed yet. Analyze one first.")
    with using_config(llm):
        result = answer_question(question, _open_store(record), k=k, overview=record.overview)
    result["repository"] = record.name
    return result


def remove(reference: str) -> RepoRecord:
    record = find_record(reference)
    if record is None:
        raise RepositoryError(f"No indexed repository matches '{reference}'.")
    delete_collection(record.collection)
    remove_record(record.key)
    return record
