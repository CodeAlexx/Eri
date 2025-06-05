"""
SD Forms - A modular Stable Diffusion interface with component-based pipelines
Backend system with FastAPI + Flutter web frontend
"""

from .main import main, initialize_backend, register_components

__version__ = "2.0.0"
__all__ = ['main', 'initialize_backend', 'register_components']