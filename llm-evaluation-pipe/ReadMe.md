# LLM Evaluation Pipeline

Automated pipeline for evaluating LLM responses in real-time based on relevance, factual accuracy, and performance metrics.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llm-evaluation-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

4. Prepare your data:
- Place conversation JSON in `data/conversation.json`
- Place sources JSON in `data/sources.json`

5. Run the pipeline:
```bash
python main.py
```

## 📋 Requirements

```txt
aiohttp==3.9.1
python-dotenv==1.0.0
```

## 🏗️ Architecture

### Overview
The pipeline uses a modular, async-first architecture:

```
Input JSONs → Data Loader → Parallel Evaluators → Score Aggregator → Results
```

### Components

1. **Main Pipeline** (`main.py`)
   - Orchestrates entire evaluation flow
   - Manages parallel execution
   - Aggregates results

2. **Evaluators**
   - `RelevanceEvaluator`: Checks if response addresses query completely
   - `HallucinationEvaluator`: Verifies factual grounding in sources
   - `PerformanceEvaluator`: Tracks latency and cost metrics

3. **Utilities**
   - `GeminiClient`: Async wrapper for Gemini API
   - `DataLoader`: JSON parsing and validation

### Design Flow

```
┌─────────────────┐
│  Load JSON Data │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Extract Query + Response + Sources │
└────────┬────────────────────────────┘
         │
         ▼
    ┌────┴────┐
    │ Async   │
    │ Parallel│
    │ Eval    │
    └────┬────┘
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
  ┌───┐┌───┐┌───┐
  │Rel││Hal││Per│
  │   ││   ││   │
  └─┬─┘└─┬─┘└─┬─┘
    └────┴────┘
         │
         ▼
   ┌──────────┐
   │Aggregate │
   │  Score   │
   └─────┬────┘
         │
         ▼
    ┌────────┐
    │ Output │
    │  JSON  │
    └────────┘
```

## 💡 Design Decisions

### Why This Architecture?

1. **Modular Evaluators**
   - Each metric is independent
   - Easy to add/remove evaluators
   - Testable in isolation
   - Follows single responsibility principle

2. **Async/Await Pattern**
   - Parallel API calls reduce latency
   - Non-blocking I/O for better throughput
   - Critical for real-time evaluation

3. **LLM-as-Judge Approach**
   - More nuanced than rule-based metrics
   - Understands semantic similarity
   - Can detect subtle issues
   - Gemini chosen for cost-effectiveness

4. **Structured Prompts**
   - Request JSON responses
   - Minimize token usage
   - Easier parsing and error handling

### Why Not Other Approaches?

**Rule-Based Metrics (Rejected)**
- BLEU/ROUGE insufficient for semantic evaluation
- Can't detect nuanced hallucinations
- Rigid, doesn't understand context

**Embedding Similarity (Not Sufficient)**
- Good supplement but incomplete
- Misses logical inconsistencies
- Can't evaluate completeness well

**Multiple LLM Calls Per Metric (Too Expensive)**
- Would increase cost 3-5x
- Higher latency
- Diminishing returns on accuracy

## 📊 Evaluation Metrics

### 1. Response Relevance (40% weight)
- Measures if response addresses user query
- Checks completeness of answer
- Considers conversation context
- Score: 0-10

### 2. Hallucination Detection (40% weight)
- Verifies claims against sources
- Identifies unsupported statements
- Checks factual grounding
- Score: 0-10 (higher = more accurate)

### 3. Performance (20% weight)
- Tracks response latency
- Estimates API costs
- Combined efficiency score
- Score: 0-10

### Overall Score
Weighted average of all metrics (0-10 scale)

## ⚡ Scalability Optimizations

### For Million-Scale Daily Operations

1. **Async Architecture**
   - Parallel evaluations reduce time by ~70%
   - Non-blocking I/O throughout pipeline
   - Connection pooling for HTTP requests

2. **Caching Strategy**
   ```python
   # Pseudo-code
   cache_key = hash(query + response + sources)
   if cache_key in redis_cache:
       return cached_result
   ```

3. **Batch Processing**
   - Group similar queries
   - Evaluate in batches of 10-50
   - Reduces API overhead

4. **Rate Limiting**
   - Exponential backoff on errors
   - Respect API quotas
   - Queue management for peak loads

5. **Prompt Optimization**
   - Minimal token usage
   - Structured outputs (JSON)
   - Truncate long contexts intelligently

6. **Smart Source Selection**
   - Limit to top 5 most relevant sources
   - Truncate long documents
   - Balance accuracy vs. cost

7. **Monitoring & Logging**
   - Track evaluation latency
   - Monitor API costs
   - Alert on anomalies

### Cost Analysis (at scale)

**Per Evaluation:**
- Relevance: ~500 tokens = $0.000075
- Hallucination: ~600 tokens = $0.00009
- Performance: 0 API calls = $0
- **Total: ~$0.000165 per evaluation**

**1M evaluations/day:**
- Daily cost: ~$165
- Monthly cost: ~$5,000
- With caching (50% hit rate): ~$2,500/month

### Latency Analysis

**Single evaluation:**
- Without parallel: ~3 seconds
- With parallel: ~1 second
- With caching: ~100ms (cache hit)

**At scale (1M/day):**
- Sequential: 34.7 days
- With 100 workers: 8.3 hours
- Realistic target: 2-3 hours with batching

## 📁 Project Structure

```
llm-evaluation-pipeline/
├── main.py                    # Entry point
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── .env                       # API keys (not committed)
├── .gitignore
├── README.md
├── data/
│   ├── conversation.json      # Input: chat history
│   └── sources.json          # Input: retrieved contexts
├── evaluators/
│   ├── __init__.py
│   ├── relevance_evaluator.py
│   ├── hallucination_evaluator.py
│   └── performance_evaluator.py
└── utils/
    ├── __init__.py
    ├── gemini_client.py       # API wrapper
    └── data_loader.py         # JSON utilities
```

## 🧪 Testing

Run with sample data:
```bash
python main.py
```

Expected output:
```
🚀 LLM Evaluation Pipeline Starting...

📁 Loading data files...
✅ Loaded 5 messages
✅ Loaded 3 source documents

⚙️  Running evaluation pipeline...

============================================================
EVALUATION RESULTS
============================================================

📊 Overall Score: 8.2/10
⏱️  Evaluation Time: 1247.32ms

Detailed Metrics:
------------------------------------------------------------

RELEVANCE
  Score: 8.5/10
  Reasoning: Response addresses query well with minor gaps

HALLUCINATION
  Score: 9.0/10
  Reasoning: All claims grounded in sources

PERFORMANCE
  Score: 7.5/10
  Reasoning: Good latency, cost-efficient

💾 Results saved to evaluation_results.json

✨ Evaluation complete!
```

## 🔧 Configuration

Edit `config.py` to adjust:
- Score weights
- API model
- Performance thresholds
- Token limits

## 📝 Input Format

### conversation.json
```json
[
  {
    "sender": "user",
    "message": "What is the capital of France?",
    "timestamp": "2024-12-10T10:00:00Z"
  },
  {
    "sender": "bot",
    "message": "The capital of France is Paris.",
    "timestamp": "2024-12-10T10:00:01Z"
  }
]
```

### sources.json
```json
[
  {
    "context": "Paris is the capital and largest city of France.",
    "source": "knowledge_base",
    "relevance_score": 0.95
  }
]
```

## 📤 Output Format

```json
{
  "overall_score": 8.2,
  "evaluation_time_ms": 1247.32,
  "metrics": {
    "relevance": {
      "score": 8.5,
      "reasoning": "...",
      "completeness": "complete"
    },
    "hallucination": {
      "score": 9.0,
      "reasoning": "...",
      "grounded": "fully"
    },
    "performance": {
      "score": 7.5,
      "latency_ms": 500,
      "estimated_cost_usd": 0.000165
    }
  }
}
```

## 🚀 Future Enhancements

- [ ] Add semantic similarity metrics
- [ ] Implement Redis caching
- [ ] Add batch processing API
- [ ] Create web dashboard
- [ ] Add more evaluation metrics
- [ ] Support multiple LLM providers
- [ ] Implement A/B testing framework

## 👤 Author

Created for BeyondChats LLM Engineer Internship Assignment

## 📄 License

This code is submitted as part of an internship application and is the property of the author.

---

**Note:** Remember to never commit your `.env` file with API keys!
