from __future__ import annotations

import argparse

from src.app import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--source-file", help="Optional source PDF filter")
    args = parser.parse_args()

    service = RagService()
    result = service.ask(args.question, source_file=args.source_file)

    print("\nAnswer:\n")
    print(result.answer)
    print("\nCitations:\n")
    for idx, citation in enumerate(result.citations, start=1):
        print(
            f"[{idx}] {citation.source_filename} p.{citation.page} ({citation.chunk_type})\n"
            f"    {citation.quote}\n"
        )


if __name__ == "__main__":
    main()
