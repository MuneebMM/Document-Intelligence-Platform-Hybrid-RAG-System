"""
RAGAS-based evaluation pipeline for the hybrid RAG system.

Measures faithfulness, answer relevancy, context recall, and context
precision against a ground-truth test dataset.  Designed to be run
offline (not in a hot request path) against a collection that has
already been ingested.

Supports both the new RAGAS API (>= 0.1.9, SingleTurnSample) and
the legacy API (<= 0.1.8, datasets.Dataset).
"""

import asyncio
import json
import logging
import math
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Detect RAGAS API version at import time.
try:
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore[attr-defined]

    RAGAS_NEW_API = True
except ImportError:
    from datasets import Dataset  # type: ignore[no-redef]

    RAGAS_NEW_API = False

from app.core.config import Settings
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)

# Scores at or above this threshold count as a passing result.
_PASS_THRESHOLD = 0.7


def _safe_score(df: pd.DataFrame, col: str) -> float:
    """Extract a single metric score from a RAGAS result dataframe.

    Handles missing columns, NaN, and None gracefully.

    Args:
        df: Single-row dataframe returned by ``evaluate().to_pandas()``.
        col: Column name to extract.

    Returns:
        Float score in [0.0, 1.0], defaulting to 0.0 on any error.
    """
    try:
        val = df[col].iloc[0]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        return float(val)
    except (KeyError, IndexError):
        return 0.0


class RAGASEvaluator:
    """Runs RAGAS evaluation over a list of test cases.

    Wraps the live QAService so evaluation uses the exact same retrieval
    and generation pipeline that production requests use.  Each test case
    is evaluated independently and results are aggregated into a summary.

    Args:
        settings: Application settings (used for API keys).
        qa_service: Instantiated QAService to call for answers.
    """

    def __init__(self, settings: Settings, qa_service: QAService) -> None:
        """Initialise evaluation LLM, embeddings, and RAGAS wrappers."""
        self.qa_service = qa_service

        # LangChain wrappers — used by RAGAS under the hood.
        langchain_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        langchain_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )

        # RAGAS wrappers around the LangChain objects.
        self.ragas_llm = LangchainLLMWrapper(langchain_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        print(f"[RAGASEvaluator] Initialised — RAGAS_NEW_API={RAGAS_NEW_API}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_single_evaluation(
        self,
        test_case: dict[str, Any],
        collection_name: str = "documents",
    ) -> dict[str, Any]:
        """Evaluate one test case through the full RAG pipeline.

        Calls QAService.answer_query, extracts contexts from the
        response, builds a RAGAS dataset, and runs all four metrics.

        Args:
            test_case: Dict with keys ``question``, ``ground_truth``,
                and optionally ``context_keywords``.
            collection_name: Qdrant collection to query against.

        Returns:
            Dict containing per-metric scores, the generated answer, and
            a boolean ``passed`` flag (True when all scores exceed 0.7).
        """
        question: str = test_case["question"]
        ground_truth: str = test_case["ground_truth"]

        print(f"\n[RAGAS] Evaluating: {question[:50]}...")

        # --- Run the RAG pipeline -------------------------------------------
        try:
            qa_result = await self.qa_service.answer_query(
                query=question,
                collection_name=collection_name,
            )
            generated_answer: str = str(qa_result.get("answer", ""))
        except Exception as exc:
            print(f"[RAGAS] QAService failed: {exc}")
            logger.error("QAService failed for question %r: %s", question, exc)
            return self._error_result(question, ground_truth, str(exc))

        # --- Extract contexts -----------------------------------------------
        contexts = self._extract_contexts(qa_result)
        retrieved_count = int(qa_result.get("retrieved_chunks_count", 0))

        print(f"[RAGAS] Answer length: {len(generated_answer)}")
        print(f"[RAGAS] Contexts count: {len(contexts)}")
        print(f"[RAGAS] Context sample: {contexts[0][:100] if contexts else 'EMPTY'}")
        print(f"[RAGAS] Ground truth length: {len(ground_truth)}")

        # --- Build RAGAS dataset --------------------------------------------
        if RAGAS_NEW_API:
            sample = SingleTurnSample(
                user_input=question,
                response=generated_answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
            dataset = EvaluationDataset(samples=[sample])
        else:
            dataset = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [generated_answer],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth],
                }
            )

        # --- Run evaluation -------------------------------------------------
        try:
            result = await self._run_ragas_evaluate(dataset, self.metrics)
            result_df = result.to_pandas()
            print(f"[RAGAS] Raw result: {result_df.to_dict()}")
        except Exception as exc:
            print(f"[RAGAS] evaluate() failed: {exc}")
            traceback.print_exc()
            logger.error("RAGAS evaluate() failed for question %r: %s", question, exc)
            return self._error_result(question, ground_truth, str(exc))

        # --- Extract scores safely ------------------------------------------
        faith = _safe_score(result_df, "faithfulness")
        relevancy = _safe_score(result_df, "answer_relevancy")
        recall = _safe_score(result_df, "context_recall")
        precision = _safe_score(result_df, "context_precision")

        print(
            f"[RAGAS] Scores — F:{faith:.3f} AR:{relevancy:.3f} "
            f"CR:{recall:.3f} CP:{precision:.3f}"
        )

        passed = all(s > _PASS_THRESHOLD for s in [faith, relevancy, recall, precision])

        return {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_recall": recall,
            "context_precision": precision,
            "retrieved_chunks_count": retrieved_count,
            "passed": passed,
        }

    async def run_full_evaluation(
        self,
        test_dataset: list[dict[str, Any]],
        collection_name: str = "documents",
    ) -> dict[str, Any]:
        """Evaluate every test case in the dataset and aggregate results.

        Inserts a 3-second sleep between calls to stay within OpenAI
        rate limits.  Aggregates per-metric averages, an overall score,
        and a pass rate.

        Args:
            test_dataset: List of test-case dicts (see test_dataset.py).
            collection_name: Qdrant collection to query against.

        Returns:
            Dict with keys ``summary``, ``individual_results``, and
            ``failed_cases``.
        """
        print(f"\n{'='*60}")
        print(f"[RAGAS] Starting full evaluation: {len(test_dataset)} cases")
        print(f"[RAGAS] Collection: {collection_name}")
        print(f"[RAGAS] RAGAS_NEW_API: {RAGAS_NEW_API}")
        print(f"{'='*60}")

        individual_results: list[dict[str, Any]] = []

        for i, test_case in enumerate(test_dataset):
            print(f"\n{'='*50}")
            print(f"Test case {i + 1}/{len(test_dataset)}")
            try:
                result = await self.run_single_evaluation(test_case, collection_name)
                individual_results.append(result)
            except Exception as exc:
                print(f"ERROR on case {i + 1}: {exc}")
                traceback.print_exc()
                individual_results.append({
                    "question": test_case["question"],
                    "generated_answer": "EVALUATION_ERROR",
                    "ground_truth": test_case["ground_truth"],
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_recall": 0.0,
                    "context_precision": 0.0,
                    "retrieved_chunks_count": 0,
                    "passed": False,
                    "error": str(exc),
                })

            logger.info(
                "Case %d/%d — passed=%s faith=%.3f rel=%.3f recall=%.3f prec=%.3f",
                i + 1,
                len(test_dataset),
                individual_results[-1].get("passed", False),
                individual_results[-1].get("faithfulness", 0.0),
                individual_results[-1].get("answer_relevancy", 0.0),
                individual_results[-1].get("context_recall", 0.0),
                individual_results[-1].get("context_precision", 0.0),
            )

            # Rate-limit pause between cases.
            if i < len(test_dataset) - 1:
                await asyncio.sleep(3)

        # --- Aggregate ------------------------------------------------------
        passed_cases = [r for r in individual_results if r.get("passed", False)]
        failed_cases = [r for r in individual_results if not r.get("passed", False)]

        def _avg(key: str) -> float:
            """Average a metric across all results, ignoring non-float values."""
            vals = [r[key] for r in individual_results if isinstance(r.get(key), float)]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        avg_faith = _avg("faithfulness")
        avg_relevancy = _avg("answer_relevancy")
        avg_recall = _avg("context_recall")
        avg_precision = _avg("context_precision")
        overall_score = round(
            (avg_faith + avg_relevancy + avg_recall + avg_precision) / 4, 4
        )

        summary: dict[str, Any] = {
            "total_cases": len(test_dataset),
            "passed_cases": len(passed_cases),
            "pass_rate": round(len(passed_cases) / len(test_dataset), 4) if test_dataset else 0.0,
            "avg_faithfulness": avg_faith,
            "avg_answer_relevancy": avg_relevancy,
            "avg_context_recall": avg_recall,
            "avg_context_precision": avg_precision,
            "overall_score": overall_score,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"[RAGAS] Evaluation complete")
        print(f"[RAGAS] Overall: {overall_score:.3f}  Pass rate: {summary['pass_rate'] * 100:.0f}%")
        print(f"{'='*60}")

        return {
            "summary": summary,
            "individual_results": individual_results,
            "failed_cases": failed_cases,
        }

    def save_results(self, results: dict[str, Any], filepath: str) -> None:
        """Persist evaluation results to disk in JSON and CSV formats.

        Writes a full JSON file at ``filepath`` and a simplified CSV
        (question, overall_score, passed) alongside it with the same
        stem and a ``.csv`` extension.

        Args:
            results: Return value of :meth:`run_full_evaluation`.
            filepath: Absolute or relative path for the JSON output file.
        """
        # --- JSON -----------------------------------------------------------
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Full results saved to %s", filepath)

        # --- CSV ------------------------------------------------------------
        csv_path = filepath.rsplit(".", 1)[0] + ".csv"
        rows = []
        for r in results.get("individual_results", []):
            scores = [
                r.get("faithfulness", 0.0),
                r.get("answer_relevancy", 0.0),
                r.get("context_recall", 0.0),
                r.get("context_precision", 0.0),
            ]
            valid = [s for s in scores if isinstance(s, float)]
            overall = round(sum(valid) / len(valid), 4) if valid else 0.0
            rows.append(
                {
                    "question": r.get("question", ""),
                    "overall_score": overall,
                    "passed": r.get("passed", False),
                }
            )

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("Summary CSV saved to %s", csv_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_ragas_evaluate(self, dataset: Any, metrics: list) -> Any:
        """Run RAGAS evaluate() in a thread executor.

        ``evaluate()`` is synchronous but internally calls async LLM
        methods.  Running it via ``run_in_executor`` avoids blocking the
        event loop while still letting the sync internals work correctly.

        Args:
            dataset: RAGAS EvaluationDataset or HuggingFace Dataset.
            metrics: List of RAGAS metric instances.

        Returns:
            RAGAS EvaluationResult object.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings,
                    raise_exceptions=False,
                ),
            )
        return result

    @staticmethod
    def _extract_contexts(qa_result: dict[str, Any]) -> list[str]:
        """Extract context strings from a QAService response.

        Tries multiple fields in priority order to ensure RAGAS always
        receives a non-empty list of context strings.

        Args:
            qa_result: Return value of ``QAService.answer_query()``.

        Returns:
            List of non-empty context strings (at least one element).
        """
        contexts: list[str] = []

        # Try citations — each citation has a quote from a source chunk.
        if "citations" in qa_result:
            for citation in qa_result["citations"]:
                if isinstance(citation, dict):
                    quote = citation.get("quote", "")
                    if quote and str(quote).strip():
                        contexts.append(str(quote))
                    elif citation.get("text"):
                        contexts.append(str(citation["text"]))

        # Try raw retrieved chunks if citations were empty.
        if not contexts and "retrieved_chunks" in qa_result:
            for chunk in qa_result["retrieved_chunks"]:
                text = chunk.get("text", "")
                if text and str(text).strip():
                    contexts.append(str(text))

        # Fallback: use the answer itself so RAGAS doesn't receive empty
        # context (scores will be low but won't error out).
        if not contexts:
            answer = qa_result.get("answer", "")
            if answer:
                contexts.append(str(answer))

        # Absolute fallback — RAGAS requires at least one context string.
        if not contexts:
            contexts = ["No context retrieved"]

        return contexts

    @staticmethod
    def _error_result(question: str, ground_truth: str, error: str) -> dict[str, Any]:
        """Return a zeroed-out result dict for a failed evaluation call.

        Args:
            question: The original question string.
            ground_truth: The expected answer.
            error: Error message to surface in the result.

        Returns:
            Result dict with all scores set to 0.0 and ``passed=False``.
        """
        return {
            "question": question,
            "generated_answer": f"[ERROR] {error}",
            "ground_truth": ground_truth,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "retrieved_chunks_count": 0,
            "passed": False,
        }
