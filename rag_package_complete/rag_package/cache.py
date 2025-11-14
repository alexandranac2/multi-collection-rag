import json
import hashlib
from pathlib import Path
from typing import Dict


class CacheManager:
    """Handles document hash caching."""
    
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = self._load()
    
    def _load(self) -> Dict:
        """Load cache from file."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get(self, key: str) -> str:
        """Get hash for a file."""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Set hash for a file."""
        self.cache[key] = value
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
