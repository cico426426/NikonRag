from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ragas import EvaluationDataset, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from src.app import RagService


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON eval file must be a list of objects.")
        return [item for item in payload if isinstance(item, dict)]

    raise ValueError("Unsupported eval file format. Use .jsonl or .json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nikon RAG with Ragas")
    parser.add_argument("--eval-set", required=True, help="Path to .jsonl/.json evaluation set")
    parser.add_argument("--source-file", help="Optional global source file filter")
    parser.add_argument("--out-json", default="data/eval/ragas_summary.json", help="Summary JSON output path")
    parser.add_argument("--out-csv", default="data/eval/ragas_details.csv", help="Detailed CSV output path")
    args = parser.parse_args()

    # Keep eval independent from LangSmith network availability.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    service = RagService()
    rows = _load_eval_rows(Path(args.eval_set).resolve())
    if not rows:
        raise SystemExit("Evaluation set is empty.")

    records: list[dict[str, Any]] = []
    has_reference_for_all = True

    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"Row {idx} is missing 'question'.")

        source_file = row.get("source_file") or args.source_file
        reference = row.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            has_reference_for_all = False
            reference = None

        rag_answer = service.ask(question, source_file=source_file)
        retrieved_docs = service.agent.retrieve_documents(question, source_file=source_file)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        sample: dict[str, Any] = {
            "user_input": question,
            "response": rag_answer.answer,
            "retrieved_contexts": retrieved_contexts,
        }
        if reference is not None:
            sample["reference"] = reference
        records.append(sample)

    dataset = EvaluationDataset.from_list(records)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
    ]
    if has_reference_for_all:
        metrics.extend([
            ContextRecall(),
            ContextPrecision(),
        ])

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=service.agent.answer_llm,
        embeddings=service.vector_index.embeddings,
        show_progress=True,
        raise_exceptions=False,
    )
    details = result.to_pandas()

    metric_names = [metric.name for metric in metrics]
    summary_scores = {}
    for name in metric_names:
        if name in details.columns:
            value = details[name].dropna().mean()
            summary_scores[name] = float(value) if value is not None else None

    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    details.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "num_samples": len(records),
                "has_reference_for_all": has_reference_for_all,
                "metrics": summary_scores,
                "details_csv": str(out_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRagas evaluation done.")
    print(f"Samples: {len(records)}")
    print(f"Summary: {json.dumps(summary_scores, ensure_ascii=False)}")
    print(f"Details CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")


if __name__ == "__main__":
    main()
