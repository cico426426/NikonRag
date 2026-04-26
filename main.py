from __future__ import annotations

import argparse

from src.app import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Nikon Multimodal RAG CLI")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--ingest", help="Optional PDF path to ingest before querying")
    parser.add_argument("--source-file", help="Optional source PDF filter during retrieval")
    args = parser.parse_args()

    service = RagService()

    if args.ingest:
        indexed = service.ingest(args.ingest)
        print(f"Indexed {indexed} chunks from {args.ingest}")

    if args.question:
        result = service.ask(args.question, source_file=args.source_file)
        print("\nAnswer:\n")
        print(result.answer)
        print("\nCitations:\n")
        for citation in result.citations:
            print(f"[C{citation.citation_id}] {citation.source_filename} p.{citation.page} - {citation.quote}")


if __name__ == "__main__":
    main()
