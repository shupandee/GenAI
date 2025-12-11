# utils/gemini_client.py
"""Gemini API client wrapper"""

import os
import aiohttp
from typing import Optional
from config import Config


class GeminiClient:
    """Async client for Google Gemini API"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.base_url = Config.GEMINI_API_BASE
        self.model = Config.GEMINI_MODEL
        
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate text using Gemini API"""
        
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Gemini API error: {error}")
                
                data = await response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]