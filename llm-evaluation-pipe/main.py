"""
LLM Evaluation Pipeline
Evaluates AI responses for relevance, accuracy, and performance
"""

import asyncio
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any

# IMPORTANT: Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

from evaluators.relevance_evaluator import RelevanceEvaluator
from evaluators.hallucination_evaluator import HallucinationEvaluator
from evaluators.performance_evaluator import PerformanceEvaluator
from utils.data_loader import load_conversation, load_sources
from config import Config


class EvaluationPipeline:
    """Main pipeline orchestrating all evaluations"""
    
    def __init__(self):
        """Initialize all evaluators"""
        self.relevance_eval = RelevanceEvaluator()
        self.hallucination_eval = HallucinationEvaluator()
        self.performance_eval = PerformanceEvaluator()
        
    async def evaluate_response(
        self,
        conversation: List[Dict],
        sources: List[Dict],
        response_time: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single AI response
        
        Args:
            conversation: Chat history with user query and AI response
            sources: Retrieved context from vector DB
            response_time: Optional response generation time
            
        Returns:
            Evaluation results with scores and metadata
        """
        start_time = time.time()
        
        # Extract last user message and AI response
        user_message = self._get_last_user_message(conversation)
        ai_response = self._get_last_ai_response(conversation)
        
        if not user_message or not ai_response:
            raise ValueError("Invalid conversation format")
        
        print(f"📝 User Query: {user_message[:100]}...")
        print(f"🤖 AI Response: {ai_response[:100]}...\n")
        
        # Run evaluations in parallel
        print("⚙️  Running evaluations in parallel...")
        tasks = [
            self.relevance_eval.evaluate(
                user_message, ai_response, conversation[:-1]
            ),
            self.hallucination_eval.evaluate(
                ai_response, sources
            ),
            self.performance_eval.evaluate(
                response_time or 0.0, ai_response
            )
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any errors
        relevance_result = results[0] if not isinstance(
            results[0], Exception
        ) else {"score": 0.0, "error": str(results[0])}
        
        hallucination_result = results[1] if not isinstance(
            results[1], Exception
        ) else {"score": 0.0, "error": str(results[1])}
        
        performance_result = results[2] if not isinstance(
            results[2], Exception
        ) else {"score": 0.0, "error": str(results[2])}
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            relevance_result.get("score", 0.0),
            hallucination_result.get("score", 0.0),
            performance_result.get("score", 0.0)
        )
        
        evaluation_time = time.time() - start_time
        
        return {
            "overall_score": overall_score,
            "evaluation_time_ms": round(evaluation_time * 1000, 2),
            "metrics": {
                "relevance": relevance_result,
                "hallucination": hallucination_result,
                "performance": performance_result
            },
            "metadata": {
                "user_query": user_message,
                "ai_response": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response,
                "num_sources": len(sources)
            }
        }
    
    def _get_last_user_message(self, conversation: List[Dict]) -> str:
        """Extract the last user message from conversation"""
        for msg in reversed(conversation):
            if msg.get("sender") in ["user", "human"]:
                return msg.get("message", "")
        return ""
    
    def _get_last_ai_response(self, conversation: List[Dict]) -> str:
        """Extract the last AI response from conversation"""
        for msg in reversed(conversation):
            if msg.get("sender") in ["bot", "assistant", "ai"]:
                return msg.get("message", "")
        return ""
    
    def _calculate_overall_score(
        self, 
        relevance: float, 
        hallucination: float, 
        performance: float
    ) -> float:
        """Calculate weighted overall score"""
        weights = Config.SCORE_WEIGHTS
        score = (
            relevance * weights["relevance"] +
            hallucination * weights["hallucination"] +
            performance * weights["performance"]
        )
        return round(score, 2)


async def main():
    """Main entry point"""
    print("=" * 70)
    print("🚀 LLM EVALUATION PIPELINE")
    print("=" * 70)
    print()
    
    # Verify API key is loaded
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found!")
        print()
        print("Please check:")
        print("1. Is there a .env file in the project root?")
        print("2. Does it contain: GEMINI_API_KEY=your_key_here")
        print("3. Is the .env file in the same directory as main.py?")
        print()
        print("Current directory:", os.getcwd())
        print(".env file exists:", os.path.exists(".env"))
        return
    
    print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Load data
    conversation_file = "data/conversation.json"
    sources_file = "data/sources.json"
    
    print(f"📁 Loading data files...")
    
    try:
        conversation = load_conversation(conversation_file)
        sources = load_sources(sources_file)
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Please run: python setup_data.py")
        print("Or download data from the assignment Google Drive link")
        return
    
    print(f"✅ Loaded {len(conversation)} messages")
    print(f"✅ Loaded {len(sources)} source documents")
    print()
    
    # Initialize pipeline
    try:
        pipeline = EvaluationPipeline()
        print("✅ Pipeline initialized")
        print()
    except Exception as e:
        print(f"❌ ERROR initializing pipeline: {e}")
        return
    
    # Run evaluation
    print("=" * 70)
    print("⚙️  RUNNING EVALUATION PIPELINE")
    print("=" * 70)
    print()
    
    try:
        results = await pipeline.evaluate_response(
            conversation=conversation,
            sources=sources,
            response_time=0.5  # Example response time in seconds
        )
        
        # Print results
        print()
        print("=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        print()
        print(f"Overall Score: {results['overall_score']}/10")
        print(f"Evaluation Time: {results['evaluation_time_ms']}ms")
        print()
        
        print("Detailed Metrics:")
        print("-" * 70)
        
        for metric_name, metric_data in results["metrics"].items():
            score = metric_data.get("score", 0.0)
            print(f"\n{metric_name.upper()}")
            print(f"  Score: {score}/10")
            
            if "reasoning" in metric_data:
                reasoning = metric_data['reasoning']
                print(f"  Reasoning: {reasoning}")
            
            if "issues" in metric_data and metric_data['issues']:
                print(f"  Issues: {', '.join(metric_data['issues'])}")
            
            if "error" in metric_data:
                print(f"  Error: {metric_data['error']}")
        
        print()
        print("-" * 70)
        
        # Save results
        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print()
        print(f"💾 Results saved to: {output_file}")
        print()
        print("=" * 70)
        print("✨ EVALUATION COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())