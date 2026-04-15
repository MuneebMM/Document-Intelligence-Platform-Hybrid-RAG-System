"""Document ingestion routes."""

import os

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.schemas import DocumentUploadResponse
from app.services.ingestion import IngestionService

router = APIRouter()

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})
_MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB


def _get_ingestion_service() -> IngestionService:
    """Instantiate IngestionService using the cached application settings.

    Keeping construction in a helper makes it easy to swap the dependency in
    tests without patching module-level globals.
    """
    return IngestionService(settings=get_settings())


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a document",
)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    """Accept a multipart file upload, parse it, and return ingestion metadata.

    Validates the file extension and size before passing the raw bytes to
    ``IngestionService``.  Business logic lives entirely in the service layer;
    this handler is responsible only for HTTP concerns (validation, status
    codes, error responses).

    Args:
        file: Multipart upload provided by the client.  Must be a ``.pdf``,
              ``.docx``, or ``.txt`` file no larger than 50 MB.

    Returns:
        ``DocumentUploadResponse`` with status, filename, chunk count, and
        a confirmation message.

    Raises:
        HTTPException 400: File extension is not supported.
        HTTPException 413: File exceeds the 50 MB size limit.
        HTTPException 500: An unexpected error occurred during ingestion.
    """
    # --- Validate extension ---
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )

    # --- Read bytes & validate size ---
    file_bytes = await file.read()

    if len(file_bytes) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(file_bytes) / (1024 * 1024):.1f} MB exceeds "
                f"the 50 MB limit."
            ),
        )

    # --- Ingest ---
    try:
        service = _get_ingestion_service()
        result = await service.ingest_document(
            file_bytes=file_bytes,
            filename=filename,
            file_type=ext,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return DocumentUploadResponse(
        status="success",
        filename=str(result["filename"]),
        total_chunks=int(result["total_chunks"]),
        message="Document ingested successfully",
    )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Document ingestion service health check",
)
async def documents_health() -> dict[str, str]:
    """Return liveness status for the document ingestion service.

    Returns:
        JSON body with keys ``status`` and ``service``.
    """
    return {"status": "ok", "service": "document-ingestion"}
