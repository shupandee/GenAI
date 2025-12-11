"""
Utilities for loading and parsing JSON data files

This module provides functions to:
- Load conversation and source data from JSON
- Validate data structure
- Save evaluation results
"""

import json
from typing import Dict, List, Any
from pathlib import Path


def load_conversation(file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversation JSON from file
    
    Args:
        file_path: Path to conversation JSON file
        
    Returns:
        List of message dictionaries with 'sender' and 'message' keys
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
        
    Example:
        conversation = load_conversation("data/conversation.json")
        for msg in conversation:
            print(f"{msg['sender']}: {msg['message']}")
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different possible JSON structures
    if isinstance(data, list):
        # Direct list of messages
        return data
    elif isinstance(data, dict):
        # Check for common keys
        if "messages" in data:
            return data["messages"]
        elif "conversation" in data:
            return data["conversation"]
        elif "data" in data:
            return data["data"]
        else:
            raise ValueError(
                f"Unexpected JSON structure. Expected 'messages', 'conversation', "
                f"or 'data' key, got: {list(data.keys())}"
            )
    else:
        raise ValueError(f"Expected list or dict, got: {type(data)}")


def load_sources(file_path: str) -> List[Dict[str, Any]]:
    """
    Load sources/context JSON from file
    
    Args:
        file_path: Path to sources JSON file
        
    Returns:
        List of source dictionaries with context/text information
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
        
    Example:
        sources = load_sources("data/sources.json")
        for source in sources:
            print(source.get('context', ''))
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different possible JSON structures
    if isinstance(data, list):
        # Direct list of sources
        return data
    elif isinstance(data, dict):
        # Check for common keys
        if "sources" in data:
            return data["sources"]
        elif "results" in data:
            return data["results"]
        elif "contexts" in data:
            return data["contexts"]
        elif "data" in data:
            return data["data"]
        else:
            raise ValueError(
                f"Unexpected JSON structure. Expected 'sources', 'results', "
                f"'contexts', or 'data' key, got: {list(data.keys())}"
            )
    else:
        raise ValueError(f"Expected list or dict, got: {type(data)}")


def validate_conversation(conversation: List[Dict]) -> bool:
    """
    Validate conversation structure
    
    Args:
        conversation: List of message dictionaries
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        
    Example:
        conversation = load_conversation("data/conversation.json")
        validate_conversation(conversation)
        print("✅ Conversation is valid")
    """
    if not conversation:
        raise ValueError("Conversation is empty")
    
    if not isinstance(conversation, list):
        raise ValueError(f"Conversation must be a list, got: {type(conversation)}")
    
    for i, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} is not a dictionary")
        
        if "sender" not in msg:
            raise ValueError(f"Message {i} missing 'sender' field")
        
        if "message" not in msg:
            raise ValueError(f"Message {i} missing 'message' field")
        
        # Warn about unusual sender values (but don't fail)
        if msg["sender"] not in ["user", "bot", "assistant", "human", "ai"]:
            print(f"⚠️  Warning: Message {i} has unusual sender: {msg['sender']}")
    
    return True


def validate_sources(sources: List[Dict]) -> bool:
    """
    Validate sources structure
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        
    Example:
        sources = load_sources("data/sources.json")
        validate_sources(sources)
        print("✅ Sources are valid")
    """
    if not sources:
        raise ValueError("Sources list is empty")
    
    if not isinstance(sources, list):
        raise ValueError(f"Sources must be a list, got: {type(sources)}")
    
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Source {i} is not a dictionary")
        
        # Check for text content field (flexible naming)
        has_content = any(
            key in source 
            for key in ["context", "text", "content", "passage", "document"]
        )
        
        if not has_content:
            raise ValueError(
                f"Source {i} missing content field "
                f"(expected 'context', 'text', 'content', 'passage', or 'document')"
            )
    
    return True


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to output file
        
    Example:
        results = {"score": 8.5, "metrics": {...}}
        save_results(results, "evaluation_results.json")
    """
    path = Path(output_path)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_path}")


def load_json(file_path: str) -> Any:
    """
    Generic JSON loader
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Generic JSON saver
    
    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation (default: 2)
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# Example usage and testing
if __name__ == "__main__":
    """
    Test the data loader functions
    
    To run: python -m utils.data_loader
    """
    print("🧪 Testing Data Loader...\n")
    
    # Test loading conversation
    try:
        print("Test 1: Loading conversation")
        conv = load_conversation("data/conversation.json")
        print(f"✅ Loaded {len(conv)} messages")
        
        validate_conversation(conv)
        print("✅ Conversation structure valid\n")
        
    except Exception as e:
        print(f"❌ Error loading conversation: {e}\n")
    
    # Test loading sources
    try:
        print("Test 2: Loading sources")
        sources = load_sources("data/sources.json")
        print(f"✅ Loaded {len(sources)} sources")
        
        validate_sources(sources)
        print("✅ Sources structure valid\n")
        
    except Exception as e:
        print(f"❌ Error loading sources: {e}\n")
    
    # Test saving results
    try:
        print("Test 3: Saving results")
        test_results = {
            "overall_score": 8.5,
            "test": True
        }
        save_results(test_results, "test_output.json")
        print("✅ Save successful\n")
        
    except Exception as e:
        print(f"❌ Error saving: {e}\n")
    
    print("✨ Testing complete!")