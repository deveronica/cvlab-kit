"""RFC 7807 compliant response helpers."""

from datetime import datetime
from typing import Any, Dict, List, Optional


def success_response(
    data: Any, meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a successful response following RFC 7807 standards.

    Args:
        data: The response data
        meta: Optional metadata (timestamp, version, etc.)

    Returns:
        Standardized success response
    """
    response = {"data": data}

    if meta is None:
        meta = {}

    # Add default metadata
    meta.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    meta.setdefault("version", "1.0")

    response["meta"] = meta
    return response


def error_response(
    title: str,
    status: int,
    detail: str,
    error_type: str = "about:blank",
    instance: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create an error response following RFC 7807 Problem Details standard.

    Args:
        title: Short, human-readable summary of the problem
        status: HTTP status code
        detail: Human-readable explanation specific to this occurrence
        error_type: URI reference that identifies the problem type
        instance: URI reference that identifies the specific occurrence
        errors: Additional error details

    Returns:
        RFC 7807 compliant error response
    """
    response = {"type": error_type, "title": title, "status": status, "detail": detail}

    if instance:
        response["instance"] = instance

    if errors:
        response["errors"] = errors

    return response


def paginated_response(
    data: List[Any],
    total: int,
    page: int = 1,
    per_page: int = 50,
    links: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a paginated response.

    Args:
        data: List of items for current page
        total: Total number of items
        page: Current page number
        per_page: Items per page
        links: Navigation links (next, prev, etc.)

    Returns:
        Paginated response with metadata
    """
    meta = {
        "total": total,
        "page": page,
        "per_page": per_page,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0",
    }

    response = {"data": data, "meta": meta}

    if links:
        response["links"] = links

    return response
