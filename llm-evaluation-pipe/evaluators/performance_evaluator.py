# evaluators/performance_evaluator.py
"""Tracks latency and cost metrics"""

from typing import Dict, Any
from config import Config


class PerformanceEvaluator:
    """Evaluates response latency and API costs"""
    
    def __init__(self):
        self.cost_per_token = Config.GEMINI_COST_PER_1K_TOKENS / 1000
        
    async def evaluate(
        self,
        response_time: float,
        ai_response: str
    ) -> Dict[str, Any]:
        """
        Evaluate performance metrics
        
        Args:
            response_time: Time taken to generate response (seconds)
            ai_response: The generated response text
            
        Returns:
            Performance scores and metrics
        """
        
        # Calculate token count (rough estimate)
        token_count = len(ai_response.split()) * 1.3  # ~1.3 tokens per word
        
        # Calculate cost
        cost = token_count * self.cost_per_token
        
        # Score based on latency (0-10 scale)
        latency_score = self._score_latency(response_time)
        
        # Score based on cost efficiency
        cost_score = self._score_cost(cost)
        
        # Combined score
        overall_score = (latency_score + cost_score) / 2
        
        return {
            "score": round(overall_score, 2),
            "latency_ms": round(response_time * 1000, 2),
            "latency_score": round(latency_score, 2),
            "estimated_tokens": int(token_count),
            "estimated_cost_usd": round(cost, 6),
            "cost_score": round(cost_score, 2),
            "reasoning": self._generate_reasoning(
                response_time, cost, latency_score, cost_score
            )
        }
    
    def _score_latency(self, response_time: float) -> float:
        """Score latency (lower is better)"""
        if response_time < 0.5:
            return 10.0
        elif response_time < 1.0:
            return 9.0
        elif response_time < 2.0:
            return 7.0
        elif response_time < 3.0:
            return 5.0
        elif response_time < 5.0:
            return 3.0
        else:
            return 1.0
    
    def _score_cost(self, cost: float) -> float:
        """Score cost efficiency"""
        if cost < 0.001:
            return 10.0
        elif cost < 0.005:
            return 8.0
        elif cost < 0.01:
            return 6.0
        elif cost < 0.02:
            return 4.0
        else:
            return 2.0
    
    def _generate_reasoning(
        self,
        latency: float,
        cost: float,
        latency_score: float,
        cost_score: float
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []
        
        if latency_score >= 9:
            parts.append("Excellent latency")
        elif latency_score >= 7:
            parts.append("Good latency")
        else:
            parts.append("High latency")
        
        if cost_score >= 8:
            parts.append("cost-efficient")
        else:
            parts.append("higher cost")
        
        return f"{parts[0]}, {parts[1]}"