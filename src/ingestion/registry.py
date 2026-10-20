"""Paper registry loader and validator.

Loads paper metadata from the registered evaluation corpus without hardcoding
or duplicating paper details.
"""

import json
from pathlib import Path
from typing import List, Union
from src.ingestion.models import DocumentMetadata


class RegistryValidationError(Exception):
    """Raised when the paper registry contains missing or invalid records."""
    pass


def load_paper_registry(registry_path: Union[str, Path]) -> List[DocumentMetadata]:
    """Load and validate the list of registered research papers from JSON.

    Args:
        registry_path: Path to paper_registry.json.

    Returns:
        List of validated DocumentMetadata objects.

    Raises:
        FileNotFoundError: If registry_path does not exist.
        RegistryValidationError: If registry JSON is malformed or lacks required fields.
    """
    path = Path(registry_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Paper registry file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RegistryValidationError(f"Invalid JSON in paper registry: {e}") from e

    if not isinstance(data, list):
        raise RegistryValidationError("Paper registry JSON must contain a top-level list of objects.")

    documents: List[DocumentMetadata] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise RegistryValidationError(f"Registry entry at index {idx} must be a JSON object.")

        # Ensure minimal required fields
        if not item.get("id"):
            raise RegistryValidationError(f"Registry entry at index {idx} is missing required 'id'.")
        if not item.get("title"):
            raise RegistryValidationError(f"Registry entry at index {idx} is missing required 'title'.")
        if not item.get("filename"):
            raise RegistryValidationError(f"Registry entry at index {idx} is missing required 'filename'.")

        try:
            doc_meta = DocumentMetadata.model_validate(item)
            documents.append(doc_meta)
        except Exception as e:
            raise RegistryValidationError(f"Validation error for paper '{item.get('id')}': {e}") from e

    return documents

