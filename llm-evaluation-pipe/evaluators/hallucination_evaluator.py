# evaluators/hallucination_evaluator.py
"""Detects hallucinations and verifies factual accuracy"""

from typing import Dict, List, Any, Optional
import json
import re
import logging
from utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """Detects hallucinations by checking against sources"""
    
    def __init__(self):
        self.client = GeminiClient()
        
    async def evaluate(
        self,
        ai_response: str,
        sources: List[Dict]
    ) -> Dict[str, Any]:
        """Check if response is grounded in provided sources"""
        
        source_text = self._extract_source_text(sources)
        
        if not source_text:
            return self._create_response(
                score=5.0,
                reasoning="No sources provided for verification",
                grounded="unknown",
                issues=["No source documents available"]
            )
        
        prompt = self._build_prompt(ai_response, source_text)
        
        try:
            result = await self.client.generate(
                prompt,
                temperature=0.0,
                max_tokens=1500
            )
            
            logger.info(f"Hallucination eval raw (first 200 chars): {result[:200]}")
            return self._parse_with_fallback(result)
            
        except Exception as e:
            logger.error(f"Hallucination evaluation error: {e}", exc_info=True)
            return self._create_response(
                score=5.0,
                reasoning=f"Evaluation error: {str(e)}",
                grounded="unknown",
                issues=[f"Error: {str(e)}"]
            )
    
    def _extract_source_text(self, sources: List[Dict]) -> str:
        """Extract text content from sources"""
        if not sources:
            return ""
        
        texts = []
        for i, source in enumerate(sources[:5], 1):
            content = source.get("context") or source.get("text", "")
            if content:
                texts.append(f"[Source {i}] {content.strip()}")
        
        return "\n\n".join(texts)[:4000]
    
    def _build_prompt(self, response: str, sources: str) -> str:
        """Build prompt that enforces JSON output"""
        
        return f"""Compare the AI response against source documents. Output ONLY valid JSON.

SOURCES:
{sources}

AI RESPONSE:
{response}

Rate accuracy 0-10. Respond with this EXACT format (replace values):

{{"score": 8.0, "reasoning": "your brief explanation here", "grounded": "partially", "issues": ["issue 1", "issue 2"], "unsupported_claims": ["claim 1"]}}

Rules:
- score: 0-10 (decimals ok)
- grounded: "fully", "partially", "not", or "unknown"
- issues/unsupported_claims: arrays (use [] if empty)

JSON only, no other text:"""
    
    def _parse_with_fallback(self, result: str) -> Dict[str, Any]:
        """Parse with multiple strategies and guaranteed fallback"""
        
        result = result.strip()
        
        # Try 5 parsing strategies in order
        strategies = [
            self._try_direct_json,
            self._try_clean_json,
            self._try_extract_json,
            self._try_repair_json,
            self._try_manual_extraction
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                parsed = strategy(result)
                if parsed:
                    logger.info(f"Strategy {i} succeeded")
                    return self._normalize(parsed)
            except Exception as e:
                logger.debug(f"Strategy {i} failed: {e}")
                continue
        
        # Absolute fallback - construct from any text patterns found
        logger.warning("All strategies failed, using text analysis")
        return self._intelligent_fallback(result)
    
    def _try_direct_json(self, text: str) -> Optional[Dict]:
        """Try parsing as direct JSON"""
        return json.loads(text)
    
    def _try_clean_json(self, text: str) -> Optional[Dict]:
        """Remove common wrappers and parse"""
        cleaned = text
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'(?i)^(json|output|response)[:\s]+', '', cleaned)
        return json.loads(cleaned.strip())
    
    def _try_extract_json(self, text: str) -> Optional[Dict]:
        """Find and extract JSON object from text"""
        # Find outermost braces
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find matching close
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    data = json.loads(json_str)
                    if "score" in data:  # Validate structure
                        return data
        return None
    
    def _try_repair_json(self, text: str) -> Optional[Dict]:
        """Attempt to fix malformed JSON"""
        # Extract JSON-like content
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if not match:
            return None
        
        fixed = match.group()
        fixed = fixed.replace("'", '"')  # Fix quotes
        fixed = re.sub(r'(\w+)\s*:', r'"\1":', fixed)  # Quote keys
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)  # Remove trailing commas
        
        return json.loads(fixed)
    
    def _try_manual_extraction(self, text: str) -> Optional[Dict]:
        """Extract fields manually with comprehensive patterns"""
        
        data = {}
        
        # Extract score - try multiple patterns
        for pattern in [
            r'"?score"?\s*[:=]\s*([0-9]+\.?[0-9]*)',
            r'\bscore\s+(?:is\s+)?([0-9]+\.?[0-9]*)',
            r'\brating\s+(?:of\s+)?([0-9]+\.?[0-9]*)'
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["score"] = float(match.group(1))
                break
        
        # Extract reasoning
        for pattern in [
            r'"?reasoning"?\s*[:=]\s*"([^"]+)"',
            r'"?reasoning"?\s*[:=]\s*\'([^\']+)\'',
            r'reasoning\s*[:=]\s*([^,\n}{]+)'
        ]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                data["reasoning"] = match.group(1).strip(' "\',')
                break
        
        # Extract grounded
        for pattern in [
            r'"?grounded"?\s*[:=]\s*"?(fully|partially|not|unknown)"?',
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["grounded"] = match.group(1).lower()
                break
        
        # Extract arrays
        for field in ["issues", "unsupported_claims"]:
            pattern = f'"?{field}"?\\s*[:=]\\s*\\[(.*?)\\]'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1)
                items = re.findall(r'["\']([^"\']+)["\']', content)
                data[field] = items
        
        # Only return if we found score (main field)
        if "score" in data:
            return data
        return None
    
    def _intelligent_fallback(self, text: str) -> Dict[str, Any]:
        """Analyze text content to build best-guess response"""
        
        text_lower = text.lower()
        
        # Guess score from text sentiment
        score = 5.0
        if any(word in text_lower for word in ["accurate", "correct", "supported", "verified", "fully"]):
            score = 8.0
        elif any(word in text_lower for word in ["mostly", "generally", "partial"]):
            score = 6.5
        elif any(word in text_lower for word in ["unsupported", "fabricated", "hallucination", "incorrect"]):
            score = 3.0
        
        # Guess grounded status
        grounded = "unknown"
        if "fully" in text_lower and ("grounded" in text_lower or "supported" in text_lower):
            grounded = "fully"
        elif "partially" in text_lower:
            grounded = "partially"
        elif "not" in text_lower and "grounded" in text_lower:
            grounded = "not"
        
        return self._create_response(
            score=score,
            reasoning="Extracted from unstructured response",
            grounded=grounded,
            issues=["Could not parse structured evaluation"],
            unsupported_claims=[]
        )
    
    def _normalize(self, data: Dict) -> Dict[str, Any]:
        """Normalize and validate parsed data"""
        
        # Validate score
        try:
            score = float(data.get("score", 5.0))
            score = max(0.0, min(10.0, score))
        except (ValueError, TypeError):
            score = 5.0
        
        # Validate grounded
        grounded = str(data.get("grounded", "unknown")).lower().strip()
        if grounded not in ["fully", "partially", "not", "unknown"]:
            grounded = "unknown"
        
        # Validate arrays
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []
        
        claims = data.get("unsupported_claims", [])
        if not isinstance(claims, list):
            claims = [str(claims)] if claims else []
        
        # Validate reasoning
        reasoning = str(data.get("reasoning", "No reasoning provided")).strip()
        if not reasoning:
            reasoning = "Evaluation completed"
        
        return self._create_response(
            score=score,
            reasoning=reasoning[:500],
            grounded=grounded,
            issues=[str(i)[:200] for i in issues][:10],
            unsupported_claims=[str(c)[:200] for c in claims][:10]
        )
    
    def _create_response(
        self,
        score: float,
        reasoning: str,
        grounded: str,
        issues: List[str] = None,
        unsupported_claims: List[str] = None
    ) -> Dict[str, Any]:
        """Create standardized response structure"""
        return {
            "score": score,
            "reasoning": reasoning,
            "grounded": grounded,
            "issues": issues or [],
            "unsupported_claims": unsupported_claims or []
        }
