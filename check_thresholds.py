import json

from src.query_expander import should_expand_query
from src.vector_store import get_vector_store, search_code

with open("data/evaluation/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

vector_store = get_vector_store()

for threshold in (0.4, 0.5, 0.6, 0.7):
    triggered = []

    for item in questions:
        question = item["question"]

        documents = search_code(
            query=question,
            vector_store=vector_store,
            k=5,
        )

        if should_expand_query(question, documents, min_overlap=threshold):
            triggered.append(item["id"])

    print(f"\nThreshold: {threshold}")
    print(f"Triggered: {len(triggered)}/{len(questions)}")
    print("Question IDs:", ", ".join(triggered))
    print("R15 triggered:", "R15" in triggered)
