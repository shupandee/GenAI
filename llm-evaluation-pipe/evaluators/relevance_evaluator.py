# evaluators/relevance_evaluator.py
"""Evaluates response relevance and completeness"""

from typing import Dict, List, Any, Optional
import json
import re
import logging
from utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """Evaluates if AI response is relevant and complete"""
    
    def __init__(self):
        self.client = GeminiClient()
        
    async def evaluate(
        self,
        user_query: str,
        ai_response: str,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate response relevance and completeness"""
        
        prompt = self._build_prompt(user_query, ai_response, conversation_history)
        
        try:
            result = await self.client.generate(
                prompt,
                temperature=0.0,
                max_tokens=1500
            )
            
            logger.info(f"Relevance eval raw (first 200 chars): {result[:200]}")
            return self._parse_with_fallback(result)
            
        except Exception as e:
            logger.error(f"Relevance evaluation error: {e}", exc_info=True)
            return self._create_response(
                score=5.0,
                reasoning=f"Evaluation error: {str(e)}",
                completeness="unknown",
                missing_aspects=[f"Error: {str(e)}"]
            )
    
    def _build_prompt(
        self,
        query: str,
        response: str,
        history: List[Dict]
    ) -> str:
        """Build prompt that enforces JSON output"""
        
        context = ""
        if history:
            recent = history[-3:]
            context = "\n".join([
                f"{msg.get('sender', 'User')}: {msg.get('message', '')}"
                for msg in recent if msg.get('message')
            ])
        
        return f"""Evaluate how well the AI response addresses the user's query. Output ONLY valid JSON.

QUERY: {query}

{f"CONTEXT:\n{context}\n" if context else ""}
RESPONSE:
{response}

Rate relevance 0-10. Respond with this EXACT format (replace values):

{{"score": 8.5, "reasoning": "your brief explanation here", "completeness": "partial", "missing_aspects": ["aspect 1", "aspect 2"]}}

Rules:
- score: 0-10 (decimals ok)
- completeness: "complete", "partial", "incomplete", or "unknown"
- missing_aspects: array (use [] if empty)

JSON only, no other text:"""
    
    def _parse_with_fallback(self, result: str) -> Dict[str, Any]:
        """Parse with multiple strategies and guaranteed fallback"""
        
        result = result.strip()
        
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
        
        logger.warning("All strategies failed, using text analysis")
        return self._intelligent_fallback(result)
    
    def _try_direct_json(self, text: str) -> Optional[Dict]:
        return json.loads(text)
    
    def _try_clean_json(self, text: str) -> Optional[Dict]:
        cleaned = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'(?i)^(json|output|response)[:\s]+', '', cleaned)
        return json.loads(cleaned.strip())
    
    def _try_extract_json(self, text: str) -> Optional[Dict]:
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    data = json.loads(text[start:i+1])
                    if "score" in data:
                        return data
        return None
    
    def _try_repair_json(self, text: str) -> Optional[Dict]:
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if not match:
            return None
        
        fixed = match.group()
        fixed = fixed.replace("'", '"')
        fixed = re.sub(r'(\w+)\s*:', r'"\1":', fixed)
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        return json.loads(fixed)
    
    def _try_manual_extraction(self, text: str) -> Optional[Dict]:
        data = {}
        
        # Extract score
        for pattern in [
            r'"?score"?\s*[:=]\s*([0-9]+\.?[0-9]*)',
            r'\bscore\s+(?:is\s+)?([0-9]+\.?[0-9]*)'
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
        
        # Extract completeness
        for pattern in [
            r'"?completeness"?\s*[:=]\s*"?(complete|partial|incomplete|unknown)"?'
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["completeness"] = match.group(1).lower()
                break
        
        # Extract missing_aspects
        pattern = r'"?missing_aspects"?\s*[:=]\s*\[(.*?)\]'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1)
            items = re.findall(r'["\']([^"\']+)["\']', content)
            data["missing_aspects"] = items
        
        if "score" in data:
            return data
        return None
    
    def _intelligent_fallback(self, text: str) -> Dict[str, Any]:
        """Analyze text to build best-guess response"""
        
        text_lower = text.lower()
        
        # Guess score
        score = 5.0
        if any(word in text_lower for word in ["comprehensive", "complete", "fully addresses", "excellent"]):
            score = 9.0
        elif any(word in text_lower for word in ["good", "relevant", "addresses"]):
            score = 7.0
        elif any(word in text_lower for word in ["partial", "some", "gaps"]):
            score = 5.5
        elif any(word in text_lower for word in ["incomplete", "missing", "off-topic"]):
            score = 3.0
        
        # Guess completeness
        completeness = "unknown"
        if "complete" in text_lower and "in" not in text_lower[:text_lower.index("complete")+20]:
            completeness = "complete"
        elif "partial" in text_lower:
            completeness = "partial"
        elif "incomplete" in text_lower:
            completeness = "incomplete"
        
        return self._create_response(
            score=score,
            reasoning="Extracted from unstructured response",
            completeness=completeness,
            missing_aspects=["Could not parse structured evaluation"]
        )
    
    def _normalize(self, data: Dict) -> Dict[str, Any]:
        """Normalize and validate parsed data"""
        
        try:
            score = float(data.get("score", 5.0))
            score = max(0.0, min(10.0, score))
        except (ValueError, TypeError):
            score = 5.0
        
        completeness = str(data.get("completeness", "unknown")).lower().strip()
        if completeness not in ["complete", "partial", "incomplete", "unknown"]:
            completeness = "unknown"
        
        missing = data.get("missing_aspects", [])
        if not isinstance(missing, list):
            missing = [str(missing)] if missing else []
        
        reasoning = str(data.get("reasoning", "No reasoning provided")).strip()
        if not reasoning:
            reasoning = "Evaluation completed"
        
        return self._create_response(
            score=score,
            reasoning=reasoning[:500],
            completeness=completeness,
            missing_aspects=[str(a)[:200] for a in missing][:10]
        )
    
    def _create_response(
        self,
        score: float,
        reasoning: str,
        completeness: str,
        missing_aspects: List[str] = None
    ) -> Dict[str, Any]:
        """Create standardized response structure"""
        return {
            "score": score,
            "reasoning": reasoning,
            "completeness": completeness,
            "missing_aspects": missing_aspects or []
        }