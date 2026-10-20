"""Pagination resolution logic for grounding page citations.

Records the printed page number ONLY when it can be verified from the document
or an explicit, verified pagination mapping.
"""

import re
from typing import Optional, Dict, Union


class PaginationResolver:
    """Resolves printed page numbers from explicit mappings or verified rules."""

    @staticmethod
    def resolve_printed_page(
        physical_page_index: int,
        mapping_spec: Optional[str] = None,
        explicit_offset: Optional[int] = None,
        explicit_map: Optional[Dict[int, int]] = None,
    ) -> Optional[int]:
        """Resolve the printed page number for a given 0-based physical page index.

        Args:
            physical_page_index: Zero-based physical page index (0, 1, 2, ...).
            mapping_spec: String specification from paper metadata, e.g.
                'physical_index_i = printed_page_i_plus_1' or
                'physical_index_i = physical_page_i_plus_1 (preprint format)'.
            explicit_offset: Numeric offset where printed = physical_index + explicit_offset.
            explicit_map: Dictionary mapping physical_index -> printed_page.

        Returns:
            The verified printed page number as an integer, or None if the document
            is a preprint without printed page numbers or has unverified pagination.
        """
        # 1. Highest precedence: explicit dictionary mapping
        if explicit_map is not None:
            return explicit_map.get(physical_page_index)

        # 2. Second precedence: explicit numeric offset
        if explicit_offset is not None:
            return physical_page_index + explicit_offset

        # 3. Third precedence: parse mapping_spec string
        if not mapping_spec:
            return None

        normalized_spec = mapping_spec.lower().strip()

        # If explicitly marked as preprint format with no printed numbers, return None
        if "preprint format" in normalized_spec or "no printed page numbers" in normalized_spec:
            return None

        # Check for standard offset pattern: 'printed_page_i_plus_(\d+)'
        match_plus = re.search(r"printed_page_i_plus_(\d+)", normalized_spec)
        if match_plus:
            offset = int(match_plus.group(1))
            return physical_page_index + offset

        # Check for offset pattern: 'printed_page_i_minus_(\d+)'
        match_minus = re.search(r"printed_page_i_minus_(\d+)", normalized_spec)
        if match_minus:
            offset = int(match_minus.group(1))
            return max(1, physical_page_index - offset)

        # Unrecognized or unverified mapping specification
        return None

