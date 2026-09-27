import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


class CacheManager:
    """SHA-256 per file, so unchanged documents are not re-embedded."""

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = self._load()

    def _load(self) -> Dict[str, str]:
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def set(self, key: str, value: str) -> None:
        self.cache[key] = value

    def remove(self, key: str) -> None:
        self.cache.pop(key, None)

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
