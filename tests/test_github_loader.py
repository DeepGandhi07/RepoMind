from src.github_loader import (
    fetch_python_files,
    parse_github_url,
)

repo_url = "https://github.com/psf/requests"
owner, repo = parse_github_url(repo_url)
print(f"Owner: {owner}")
print(f"Repository: {repo}")

files = fetch_python_files(repo_url)
print(f"\\nFound {len(files)} Python files.\\n")

for file in files[:10]:
    print(f"- {file['path']}")
