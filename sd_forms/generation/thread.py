"""
Generation thread - Stub for backend compatibility
Note: Generation now handled by FastAPI backend with async execution
"""

import asyncio
# PyQt5 functionality removed - using web backend now

from ..core import Pipeline, ProcessContext, ComponentStatus


class GenerationThread:
    """Generation thread stub for backend compatibility"""
    
    def __init__(self, pipeline: Pipeline):
        print("ℹ️ GenerationThread: Pipeline execution moved to FastAPI backend")
        self.pipeline = pipeline
        self.is_running = False
        
    def start(self):
        """Start generation - stub for compatibility"""
        print("ℹ️ GenerationThread: Would start pipeline execution")
        self.is_running = True
        
    def stop(self):
        """Stop generation - stub for compatibility"""
        print("ℹ️ GenerationThread: Would stop pipeline execution")
        self.is_running = False
        
    def wait(self):
        """Wait for completion - stub for compatibility"""
        print("ℹ️ GenerationThread: Would wait for completion")
        
    def is_finished(self):
        """Check if finished - stub for compatibility"""
        return not self.is_running


# Keep import compatibility for backend
__all__ = ['GenerationThread']