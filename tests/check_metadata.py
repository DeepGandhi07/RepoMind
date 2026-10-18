from src.vector_store import get_vector_store

vector_store = get_vector_store()

results = vector_store.get(
    where={"source": "src/requests/auth.py"},
    include=["documents", "metadatas"],
)

print("Chunks found:", len(results["ids"]))

for i, (chunk_id, metadata, document) in enumerate(
    zip(results["ids"], results["metadatas"], results["documents"])
):
    print(f"\nChunk {i + 1}")
    print("ID:", chunk_id)
    print("Metadata:", metadata)
    print("Preview:", document[:100].replace("\n", " "))
    if i == 2:
        break
