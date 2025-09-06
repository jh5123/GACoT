"""
Cache Management
Handle response caching for efficient evaluation
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any


class CacheManager:
    """Manage cached LLM responses."""
    
    def __init__(self, model: str, cache_dir: str = "cache"):
        """
        Initialize cache manager.
        
        Args:
            model: Model name for cache organization
            cache_dir: Base cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create model-specific directory
        self.model_dir = self.cache_dir / model.replace("/", "_")
        self.model_dir.mkdir(exist_ok=True)
        
        self.cache_file = self.model_dir / "responses.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load existing cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"  Failed to save cache: {e}")
    
    def _get_key(self, prompt: str) -> str:
        """
        Generate cache key for prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Hash key for caching
        """
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """
        Get cached response for prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Cached response or None
        """
        key = self._get_key(prompt)
        return self.cache.get(key)
    
    def save(self, prompt: str, response: str) -> None:
        """
        Save response to cache.
        
        Args:
            prompt: Input prompt
            response: LLM response
        """
        key = self._get_key(prompt)
        self.cache[key] = response
        self._save_cache()
    
    def has_cache(self) -> bool:
        """Check if cache exists."""
        return bool(self.cache)
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache = {}
        self._save_cache()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "cache_size_bytes": self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            "model_dir": str(self.model_dir),
        }