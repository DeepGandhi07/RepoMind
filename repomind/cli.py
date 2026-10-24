"""Command-line interface.

    repomind analyze <url-or-folder>
    repomind ask "<question>" [--repo NAME] [-k N]
    repomind repos
    repomind remove <repo>
"""

import argparse
import sys

from repomind import __version__
from repomind.llm import LLM_MODEL, LLM_PROVIDER, LLMUnavailableError, ModelUnavailableError, list_models
from repomind.service import analyze, ask, list_repositories, remove


def _print_sources(sources: list[dict]) -> None:
    if not sources:
        return
    print("\nSources:")
    for source in sources:
        location = source["file"]
        if "start_line" in source and "end_line" in source:
            location += f" (lines {source['start_line']}-{source['end_line']})"
        print(f"  - {location}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repomind",
        description="Ask questions about any source repository, with cited answers.",
    )
    parser.add_argument("--version", action="version", version=f"repomind {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = commands.add_parser("analyze", help="index a repository (URL or local folder)")
    analyze_cmd.add_argument("source", help="https URL, e.g. https://github.com/psf/requests, or a local folder")
    analyze_cmd.add_argument("--token", help="access token for private repositories (or use GITHUB_TOKEN / GITLAB_TOKEN)")

    ask_cmd = commands.add_parser("ask", help="ask a question about an indexed repository")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--repo", help="repository name, key, or URL (default: most recently analyzed)")
    ask_cmd.add_argument("-k", type=int, default=5, help="number of code chunks to retrieve")

    commands.add_parser("repos", help="list indexed repositories")

    commands.add_parser("models", help="list the models your API key can use")

    remove_cmd = commands.add_parser("remove", help="delete an indexed repository")
    remove_cmd.add_argument("repo", help="repository name, key, or URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "analyze":
            print(f"Analyzing {args.source} …")
            report = analyze(
                args.source,
                token=args.token,
                on_step=lambda message: print(f"  {message}"),
            )
            record = report.record
            print(
                f"Indexed {record.name}: {record.files} files, {record.chunks} chunks "
                f"({report.chunks_changed} new or updated)."
            )
            if record.skipped:
                skipped = ", ".join(f"{count} {reason}" for reason, count in record.skipped.items())
                print(f"  Skipped: {skipped}")

        elif args.command == "ask":
            result = ask(args.question, repo=args.repo, k=args.k)
            print(result["answer"])
            _print_sources(result.get("sources", []))
            status = (result.get("grounding") or {}).get("overall_status")
            if status and status != "skipped":
                print(f"\nGrounding check: {status}")

        elif args.command == "repos":
            records = list_repositories()
            if not records:
                print("No repositories indexed yet. Run `repomind analyze <url>`.")
            for record in records:
                languages = ", ".join(list(record.languages)[:4])
                print(f"{record.name}  [{record.key}]")
                print(f"  files: {record.files}   chunks: {record.chunks}   languages: {languages}")
                print(f"  source: {record.source}   indexed: {record.indexed_at}")

        elif args.command == "models":
            print(f"Provider: {LLM_PROVIDER}   active model: {LLM_MODEL}")
            print("Models available to your key:")
            for model_id in list_models():
                marker = "  <- active" if model_id == LLM_MODEL else ""
                print(f"  {model_id}{marker}")

        elif args.command == "remove":
            record = remove(args.repo)
            print(f"Removed {record.name}.")

    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except LLMUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
