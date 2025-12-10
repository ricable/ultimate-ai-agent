import sys
import time
from typing import Optional
from datetime import datetime

class ProgressReporter:
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.last_update = 0

    def start(self, message: str) -> None:
        """Start progress tracking"""
        self.start_time = datetime.now()
        self.current_step = 0
        self._update_progress(message)

    def update(self, step: int, message: str) -> None:
        """Update progress"""
        self.current_step = min(step, self.total_steps)
        self._update_progress(message)

    def finish(self, message: str = "Complete!") -> None:
        """Mark progress as complete"""
        self.current_step = self.total_steps
        self._update_progress(message)
        print()  # New line after completion

    def _update_progress(self, message: str) -> None:
        """Update progress bar"""
        current_time = time.time()
        # Always update on message change
        if message != getattr(self, '_last_message', None) or current_time - self.last_update >= 0.1:
            self.last_update = current_time
            self._last_message = message
        percentage = (self.current_step / self.total_steps) * 100
        bar_length = 40
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        elapsed = datetime.now() - self.start_time if self.start_time else None
        time_str = f" [{str(elapsed).split('.')[0]}]" if elapsed else ""
        
        sys.stdout.write(f'\r{message} |{bar}| {percentage:>3.0f}%{time_str}')
        sys.stdout.flush()
