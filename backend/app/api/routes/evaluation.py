"""
Evaluation endpoints for running RAGAS-based RAG quality assessment.

NOTE: The POST /evaluation/run endpoint is intentionally long-running.
Evaluating 15 test cases against a live LLM pipeline takes 5–10 minutes
due to multiple OpenAI calls per case and the 2-second inter-case sleep
used to respect rate limits.  Do not call this from a synchronous context
without an appropriate HTTP timeout.
"""

import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.evaluation.test_dataset import GDPR_TEST_DATASET
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunEvaluationRequest(BaseModel):
    """Request body for the full evaluation run endpoint."""

    collection_name: str = Field(
        default="documents", description="Qdrant collection to evaluate against."
    )
    save_results: bool = Field(
        default=True, description="Persist results to disk as JSON + CSV."
    )
    results_path: str = Field(
        default="evaluation_results",
        description="Directory to write result files into.",
    )


class RunSingleRequest(BaseModel):
    """Request body for evaluating a single custom question."""

    question: str = Field(..., min_length=1, description="Question to evaluate.")
    ground_truth: str = Field(..., min_length=1, description="Expected correct answer.")
    collection_name: str = Field(
        default="documents", description="Qdrant collection to evaluate against."
    )


class ResultFileSummary(BaseModel):
    """Metadata entry for a saved evaluation result file."""

    filename: str = Field(..., description="JSON filename on disk.")
    timestamp: str = Field(..., description="ISO-8601 evaluation timestamp from summary.")
    overall_score: float = Field(..., description="Aggregate score from the summary.")


class ListResultsResponse(BaseModel):
    """Response body for GET /evaluation/results."""

    results: list[ResultFileSummary] = Field(
        ..., description="Metadata for every saved evaluation file."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_qa_service() -> QAService:
    """Instantiate a QAService from application settings.

    QAService builds its own dependencies internally and defers BM25
    index loading to the first query call via ``_ensure_ready``.

    Returns:
        A QAService instance (lazy-initialised on first query).
    """
    settings = get_settings()
    return QAService(settings=settings)


def _build_evaluator() -> RAGASEvaluator:
    """Instantiate a RAGASEvaluator backed by a live QAService.

    Returns:
        A RAGASEvaluator instance ready to run.
    """
    settings = get_settings()
    qa_service = _build_qa_service()
    return RAGASEvaluator(settings=settings, qa_service=qa_service)


def _ensure_results_dir(path: str) -> Path:
    """Create the results directory if it does not already exist.

    Args:
        path: Relative or absolute path string for the directory.

    Returns:
        Resolved Path object for the directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", tags=["evaluation"])
async def run_evaluation(request: RunEvaluationRequest) -> dict:
    """Run RAGAS evaluation over the full GDPR test dataset.

    Initialises a RAGASEvaluator backed by the live QA pipeline,
    runs all 15 test cases sequentially with a 2-second inter-case
    delay, and optionally persists the results to disk.

    NOTE: This endpoint takes 5–10 minutes for 15 test cases due to
    multiple LLM calls per case.  Set your HTTP client timeout accordingly.

    Args:
        request: Configuration for the evaluation run.

    Returns:
        Full evaluation results including summary, individual results,
        and failed cases, plus response_time_ms.
    """
    start_ms = time.time()
    logger.info(
        "Starting full evaluation run — collection=%s save=%s",
        request.collection_name,
        request.save_results,
    )

    evaluator = _build_evaluator()
    test_dataset = GDPR_TEST_DATASET

    results = await evaluator.run_full_evaluation(
        test_dataset=test_dataset,
        collection_name=request.collection_name,
    )

    if request.save_results:
        results_dir = _ensure_results_dir(request.results_path)
        timestamp = results["summary"]["evaluation_timestamp"].replace(":", "-").replace("+", "-")
        filename = f"eval_{timestamp}.json"
        filepath = str(results_dir / filename)
        evaluator.save_results(results, filepath)
        results["saved_to"] = filepath

    results["response_time_ms"] = int((time.time() - start_ms) * 1000)
    return results


@router.post("/run-single", tags=["evaluation"])
async def run_single_evaluation(request: RunSingleRequest) -> dict:
    """Evaluate a single custom question through the full RAG pipeline.

    Useful for ad-hoc quality checks without running the full dataset.

    Args:
        request: A question, its ground-truth answer, and the collection
            to evaluate against.

    Returns:
        Single-case result dict with per-metric scores and a passed flag.
    """
    start_ms = time.time()
    logger.info("Single evaluation — question=%r", request.question)

    evaluator = _build_evaluator()
    test_case = {
        "question": request.question,
        "ground_truth": request.ground_truth,
    }
    result = await evaluator.run_single_evaluation(test_case)
    result["response_time_ms"] = int((time.time() - start_ms) * 1000)
    return result


@router.get("/results", response_model=ListResultsResponse, tags=["evaluation"])
async def list_results(results_path: str = "evaluation_results") -> ListResultsResponse:
    """List all saved evaluation result files with summary metadata.

    Args:
        results_path: Directory to scan for JSON result files.

    Returns:
        List of filename, timestamp, and overall_score for each file.
    """
    directory = Path(results_path)
    if not directory.exists():
        return ListResultsResponse(results=[])

    summaries: list[ResultFileSummary] = []
    for json_file in sorted(directory.glob("eval_*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            summary = data.get("summary", {})
            summaries.append(
                ResultFileSummary(
                    filename=json_file.name,
                    timestamp=summary.get("evaluation_timestamp", ""),
                    overall_score=float(summary.get("overall_score", 0.0)),
                )
            )
        except Exception as exc:
            logger.warning("Skipping unreadable result file %s: %s", json_file.name, exc)

    return ListResultsResponse(results=summaries)


@router.get("/results/{filename}", tags=["evaluation"])
async def get_result(
    filename: str,
    results_path: str = "evaluation_results",
) -> dict:
    """Load and return a specific saved evaluation result file.

    Args:
        filename: JSON filename (e.g. ``eval_2024-01-01T12-00-00.json``).
        results_path: Directory containing the file.

    Returns:
        Full evaluation result dict as stored on disk.

    Raises:
        HTTPException 404: If the file does not exist.
        HTTPException 400: If the filename contains path traversal characters.
    """
    # Guard against path traversal attacks.
    if os.sep in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = Path(results_path) / filename
    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file '{filename}' not found in '{results_path}'.",
        )

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to read result file %s: %s", filename, exc)
        raise HTTPException(status_code=500, detail="Failed to read result file.")
