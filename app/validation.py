"""Input rules for anything that becomes a path on disk or a Chroma collection name."""

import re
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

# Chroma's own rule (3-63 chars, alphanumeric edges), narrowed to characters
# that are also safe as a single directory name: no dots, slashes or "..".
_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]$")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def validate_collection_name(name: str) -> str:
    if not _COLLECTION_NAME.fullmatch(name):
        raise ValueError(
            "collection name must be 3-63 characters of letters, digits, '_' or '-', "
            "starting and ending with a letter or digit"
        )
    return name


CollectionName = Annotated[str, AfterValidator(validate_collection_name)]


def safe_filename(filename: str) -> str:
    """Strip any client-supplied directory parts and unusual characters."""
    name = Path(filename.replace("\\", "/")).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).lstrip(".")
    return name[:120] or "upload"


def ensure_within(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and refuse anything that escapes `base` (defence in depth)."""
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError("path escapes the documents directory")
    return resolved
