import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class SecurityCache:
    def __init__(self, cache_dir: str = ".security_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.cache_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_scan_results(self, scan_id: str, results: Dict[str, Any]) -> None:
        """Save scan results to cache"""
        # Overwrite existing results for the same scan_id
        result_file = self.results_dir / f"{scan_id}_latest.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)

    def get_scan_results(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached scan results by scan_id"""
        result_file = self.results_dir / f"{scan_id}_latest.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    if 'timestamp' in data and 'results' in data:
                        return data['results']
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def clear_old_results(self, days: int = 30) -> None:
        """Clear results older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        for file in self.results_dir.glob("*.json"):
            if file.stat().st_mtime < cutoff:
                file.unlink()
