from __future__ import annotations

import argparse

from src.app import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma index from a PDF")
    parser.add_argument("--pdf", required=False, help="PDF path; default uses DEFAULT_PDF_PATH")
    args = parser.parse_args()

    service = RagService()
    pdf_path = args.pdf or service.settings.default_pdf_path
    if not pdf_path:
        raise SystemExit("No PDF provided. Use --pdf or set DEFAULT_PDF_PATH.")

    count = service.ingest(str(pdf_path))
    print(f"Indexed {count} chunks from: {pdf_path}")


if __name__ == "__main__":
    main()
