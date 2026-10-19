from google import genai
import sys
from src.answer_generator import GEMINI_MODEL
from src.grounding_checker import check_answer_grounding


def main():
    client = genai.Client()

    cases = [
        {
            "name": "Supported claim",
            "context": (
                "def __call__(self, r):\n"
                '    r.headers["Authorization"] = "Basic abc"\n'
                "    return r\n"
            ),
            "answer": (
                "The handler sets the Authorization header "
                "to a Basic authentication value."
            ),
            "expected": "supported",
        },
        {
            "name": "Contradicted claim",
            "context": (
                "def __call__(self, r):\n"
                '    r.headers["Authorization"] = "Basic abc"\n'
                '    assert "Proxy-Authorization" not in r.headers\n'
                "    return r\n"
            ),
            "answer": ("The handler sets the Proxy-Authorization header."),
            "expected": "unsupported",
        },
        {
            "name": "Insufficient evidence",
            "context": (
                "def __call__(self, r):\n"
                '    r.headers["Authorization"] = "Basic abc"\n'
                "    return r\n"
            ),
            "answer": (
                "The handler retries the request after receiving "
                "a 401 Unauthorized response."
            ),
            "expected": "insufficient_evidence",
        },
    ]
    failures = 0
    for case in cases:
        print(f"\n=== {case['name']} ===")

        result = check_answer_grounding(
            answer=case["answer"],
            context=case["context"],
            client=client,
            model=GEMINI_MODEL,
        )
        actual_statuses = {claim["status"] for claim in result["claims"]}
        passed = case["expected"] in actual_statuses
        print("Evaluation:", "PASS" if passed else "FAIL")

        if not passed:
            failures += 1

        for claim in result["claims"]:
            print("Claim:", claim["claim"])
            print("Status:", claim["status"])
            print("Evidence:", claim["evidence"])
            print("Reason:", claim["reason"])

        passed = case["expected"] in actual_statuses
        print("Evaluation:", "PASS" if passed else "REVIEW")

    print("\nThese are diagnostic evaluations, not a proof " "of grounding accuracy.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
