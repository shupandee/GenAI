"""
Utils Package - Utility Modules for LLM Evaluation Pipeline

This package contains utility modules for:
- Gemini API client (gemini_client.py)
- Data loading and parsing (data_loader.py)

Usage:
    from utils import GeminiClient, load_conversation, load_sources
    
    # Initialize API client
    client = GeminiClient()
    
    # Load data
    conversation = load_conversation("data/conversation.json")
    sources = load_sources("data/sources.json")
"""

from .gemini_client import GeminiClient
from .data_loader import (
    load_conversation,
    load_sources,
    validate_conversation,
    validate_sources,
    save_results
)

# Package metadata
__version__ = "1.0.0"
__author__ = "BeyondChats Internship Candidate"

# Public API - explicitly define what gets imported with "from utils import *"
__all__ = [
    # Gemini API client
    'GeminiClient',
    
    # Data loading functions
    'load_conversation',
    'load_sources',
    
    # Data validation functions
    'validate_conversation',
    'validate_sources',
    
    # Results output functions
    'save_results'
]