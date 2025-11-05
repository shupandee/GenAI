"""
Evaluation module for RAG AI Agent
Implements multiple evaluation metrics including RAGAs-style and LLM-as-Judge
Uses Gemini for LLM-as-judge evaluations
"""

import os
from typing import List, Dict
import json
from datetime import datetime
import numpy as np
from rag_agent import RAGAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGEvaluator:
    """Comprehensive evaluator for RAG systems"""
    
    def __init__(self, 
                 agent: RAGAgent, 
                 gemini_api_key: str = None,
                 judge_model: str = "gemini-1.5-flash"):
        """
        Initialize evaluator
        
        Args:
            agent: RAGAgent instance to evaluate
            gemini_api_key: Google Gemini API key (if None, reads from env)
            judge_model: Gemini model to use for judge evaluation
        """
        self.agent = agent
        
        # Load API key from environment if not provided
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in .env file or pass as parameter.")
        
        # Initialize Gemini for judge evaluation
        self.judge_llm = ChatGoogleGenerativeAI(
            model=judge_model,
            google_api_key=self.gemini_api_key,
            temperature=0.3,
            max_output_tokens=512,
        )
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        print(f"[EVAL] Evaluator initialized with {judge_model}")
    
    # ========================================================================
    # RAGAS-STYLE METRICS
    # ========================================================================
    
    def evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if answer is faithful to retrieved contexts
        Similar to RAGAs faithfulness metric
        """
        if not contexts:
            return 0.5
        
        context_text = "\n\n".join(contexts)
        
        prompt = f"""Context:
{context_text}

Answer: {answer}

Question: Is the answer faithful to and supported by the context? Does it contain information not present in the context?

Rate faithfulness from 0 to 1, where:
- 1.0 = Completely faithful, all claims are supported
- 0.5 = Partially faithful
- 0.0 = Not faithful, contains unsupported claims

Provide only a numeric score (0.0 to 1.0):"""
        
        try:
            response = self.judge_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract numeric score
            import re
            score_match = re.search(r'(\d*\.?\d+)', response_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            print(f"[EVAL] Faithfulness error: {e}")
            return 0.5
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate if answer is relevant to the question
        Similar to RAGAs answer relevancy metric
        """
        prompt = f"""Question: {question}

Answer: {answer}

Rate how relevant this answer is to the question from 0 to 1, where:
- 1.0 = Highly relevant, directly answers the question
- 0.5 = Partially relevant
- 0.0 = Not relevant

Provide only a numeric score (0.0 to 1.0):"""
        
        try:
            response = self.judge_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            import re
            score_match = re.search(r'(\d*\.?\d+)', response_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            print(f"[EVAL] Answer relevancy error: {e}")
            return 0.5
    
    def evaluate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate if retrieved contexts are relevant to question
        Similar to RAGAs context relevancy metric
        """
        if not contexts:
            return 0.0
        
        relevant_count = 0
        
        for ctx in contexts[:3]:  # Evaluate top 3
            prompt = f"""Question: {question}

Context: {ctx}

Is this context relevant for answering the question?
Answer with only 'YES' or 'NO':"""
            
            try:
                response = self.judge_llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                if 'yes' in response_text.lower()[:10]:
                    relevant_count += 1
            except:
                pass
        
        return relevant_count / len(contexts[:3])
    
    # ========================================================================
    # LLM-AS-JUDGE METRICS
    # ========================================================================
    
    def llm_as_judge(self, question: str, answer: str, reference: str = None) -> Dict[str, float]:
        """
        Comprehensive LLM-as-Judge evaluation using Gemini
        
        Returns:
            Dict with scores for accuracy, completeness, clarity, helpfulness
        """
        reference_text = f"\nReference Answer: {reference}" if reference else ""
        
        prompt = f"""Question: {question}

Answer: {answer}{reference_text}

Evaluate this answer on the following criteria (rate each 0-10):
1. Accuracy: Is the information correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it clear and well-structured?
4. Helpfulness: Is it useful to the user?

Provide scores in this exact format:
Accuracy: X/10
Completeness: X/10
Clarity: X/10
Helpfulness: X/10

Your evaluation:"""
        
        try:
            response = self.judge_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse scores
            scores = {}
            import re
            
            for criterion in ['Accuracy', 'Completeness', 'Clarity', 'Helpfulness']:
                # Look for pattern like "Accuracy: 8/10" or "Accuracy: 8"
                pattern = rf'{criterion}\s*:\s*(\d+)'
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    scores[criterion.lower()] = min(score, 10) / 10.0
                else:
                    scores[criterion.lower()] = 0.5
            
            # Ensure all scores present
            for criterion in ['accuracy', 'completeness', 'clarity', 'helpfulness']:
                if criterion not in scores:
                    scores[criterion] = 0.5
            
            return scores
            
        except Exception as e:
            print(f"[EVAL] LLM-as-Judge error: {e}")
            return {
                'accuracy': 0.5,
                'completeness': 0.5,
                'clarity': 0.5,
                'helpfulness': 0.5
            }
    
    # ========================================================================
    # TRADITIONAL NLP METRICS
    # ========================================================================
    
    def evaluate_rouge(self, answer: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, answer)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure,
        }
    
    def evaluate_bertscore(self, answer: str, reference: str) -> Dict[str, float]:
        """Calculate BERTScore"""
        try:
            P, R, F1 = bert_score([answer], [reference], lang='en', verbose=False)
            
            return {
                'bert_precision': P.item(),
                'bert_recall': R.item(),
                'bert_f1': F1.item()
            }
        except Exception as e:
            print(f"[EVAL] BERTScore error: {e}")
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
    
    # ========================================================================
    # COMPREHENSIVE EVALUATION
    # ========================================================================
    
    def evaluate_query(self, 
                       question: str, 
                       reference_answer: str = None,
                       include_traditional_metrics: bool = False) -> Dict:
        """
        Perform comprehensive evaluation on a single query
        
        Args:
            question: User question
            reference_answer: Ground truth answer (optional)
            include_traditional_metrics: Whether to compute ROUGE/BERTScore
            
        Returns:
            Dict with all evaluation metrics
        """
        print(f"\n[EVAL] Evaluating: {question[:50]}...")
        
        # Get agent response
        result = self.agent.query(question)
        answer = result['answer']
        docs = [doc.page_content for doc in result.get('retrieved_docs', [])]
        
        evaluation = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'num_docs_retrieved': len(docs)
        }
        
        # RAGAs-style metrics
        print("[EVAL] Computing RAGAs-style metrics...")
        evaluation['faithfulness'] = self.evaluate_faithfulness(question, answer, docs)
        evaluation['answer_relevancy'] = self.evaluate_answer_relevancy(question, answer)
        evaluation['context_relevancy'] = self.evaluate_context_relevancy(question, docs)
        
        # LLM-as-Judge
        print("[EVAL] Running Gemini LLM-as-Judge evaluation...")
        judge_scores = self.llm_as_judge(question, answer, reference_answer)
        evaluation.update(judge_scores)
        
        # Traditional metrics (if reference provided)
        if reference_answer and include_traditional_metrics:
            print("[EVAL] Computing traditional NLP metrics...")
            rouge_scores = self.evaluate_rouge(answer, reference_answer)
            evaluation.update(rouge_scores)
            
            bert_scores = self.evaluate_bertscore(answer, reference_answer)
            evaluation.update(bert_scores)
        
        # Overall score
        evaluation['overall_score'] = np.mean([
            evaluation['faithfulness'],
            evaluation['answer_relevancy'],
            evaluation['context_relevancy'],
            judge_scores['accuracy'],
            judge_scores['completeness']
        ])
        
        print(f"[EVAL] ✓ Overall Score: {evaluation['overall_score']:.3f}")
        
        return evaluation
    
    def evaluate_dataset(self, 
                        test_cases: List[Dict[str, str]],
                        output_file: str = "evaluation_results.json") -> pd.DataFrame:
        """
        Evaluate agent on multiple test cases
        
        Args:
            test_cases: List of dicts with 'question' and optionally 'reference_answer'
            output_file: Where to save results
            
        Returns:
            DataFrame with all results
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING {len(test_cases)} TEST CASES")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[EVAL] Test Case {i}/{len(test_cases)}")
            
            question = test_case['question']
            reference = test_case.get('reference_answer')
            
            try:
                eval_result = self.evaluate_query(
                    question=question,
                    reference_answer=reference,
                    include_traditional_metrics=(reference is not None)
                )
                results.append(eval_result)
            except Exception as e:
                print(f"[EVAL] Error evaluating test case {i}: {e}")
                # Add failed result
                results.append({
                    'question': question,
                    'answer': 'ERROR',
                    'overall_score': 0.0,
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[EVAL] ✓ Results saved to {output_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        metrics = ['faithfulness', 'answer_relevancy', 'context_relevancy', 
                   'accuracy', 'completeness', 'clarity', 'helpfulness', 'overall_score']
        
        for metric in metrics:
            if metric in df.columns:
                mean_score = df[metric].mean()
                std_score = df[metric].std()
                print(f"{metric.replace('_', ' ').title():.<30} {mean_score:.3f} (±{std_score:.3f})")
        
        return df
    
    def generate_report(self, results_df: pd.DataFrame, output_file: str = "evaluation_report.md"):
        """
        Generate a detailed markdown evaluation report
        
        Args:
            results_df: DataFrame with evaluation results
            output_file: Where to save the report
        """
        report = []
        report.append("# RAG System Evaluation Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Total Test Cases:** {len(results_df)}\n\n")
        
        report.append("## Overall Performance\n")
        
        metrics = ['faithfulness', 'answer_relevancy', 'context_relevancy', 
                   'accuracy', 'completeness', 'clarity', 'helpfulness', 'overall_score']
        
        report.append("| Metric | Mean | Std Dev | Min | Max |\n")
        report.append("|--------|------|---------|-----|-----|\n")
        
        for metric in metrics:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                report.append(f"| {metric.replace('_', ' ').title()} | {mean_val:.3f} | {std_val:.3f} | {min_val:.3f} | {max_val:.3f} |\n")
        
        report.append("\n## Detailed Results\n\n")
        
        for idx, row in results_df.iterrows():
            report.append(f"### Test Case {idx + 1}\n\n")
            report.append(f"**Question:** {row['question']}\n\n")
            report.append(f"**Answer:** {row['answer']}\n\n")
            report.append(f"**Overall Score:** {row.get('overall_score', 0):.3f}\n\n")
            
            report.append("**Metrics:**\n")
            for metric in ['faithfulness', 'answer_relevancy', 'context_relevancy', 'accuracy', 'completeness']:
                if metric in row:
                    report.append(f"- {metric.replace('_', ' ').title()}: {row[metric]:.3f}\n")
            
            report.append("\n---\n\n")
        
        # Write report
        with open(output_file, 'w') as f:
            f.writelines(report)
        
        print(f"[EVAL] ✓ Detailed report saved to {output_file}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize agent
    agent = RAGAgent(
        gemini_model="gemini-2.5-flash",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    agent.load_documents("./data")
    agent.build_graph()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        agent=agent,
        judge_model="gemini-2.5-flash"
    )
    
    # Define test cases
    test_cases = [
        {
            "question": "What are the benefits of renewable energy?",
            "reference_answer": "Renewable energy benefits include reduced greenhouse gas emissions, energy independence, job creation, and sustainable resource use."
        },
        {
            "question": "How does solar power work?",
            "reference_answer": "Solar power works by converting sunlight into electricity using photovoltaic cells in solar panels."
        },
        {
            "question": "What are the main challenges in renewable energy adoption?",
            "reference_answer": "Main challenges include high initial costs, intermittency issues, storage limitations, and infrastructure requirements."
        }
    ]
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(test_cases)
    
    # Generate detailed report
    evaluator.generate_report(results_df)
    
    # Display results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    display_cols = ['question', 'overall_score', 'faithfulness', 'answer_relevancy', 'accuracy']
    available_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df[available_cols].to_string())