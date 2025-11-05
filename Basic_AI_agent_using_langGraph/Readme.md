# RAG Q&A AI Agent with LangGraph & Gemini

A production-ready Retrieval-Augmented Generation (RAG) system powered by Google Gemini, LangGraph, and Streamlit. Features a 4-node workflow with intelligent planning, retrieval, answer generation, and self-reflection.

## 🌟 Features

- **4-Node LangGraph Workflow**: Plan → Retrieve → Answer → Reflect
- **Google Gemini Integration**: Advanced LLM for generation and evaluation
- **Vector Search**: ChromaDB with HuggingFace embeddings
- **Interactive UI**: Beautiful Streamlit interface
- **Comprehensive Evaluation**: RAGAs-style metrics, LLM-as-Judge, ROUGE, BERTScore
- **Multi-format Support**: Load `.txt` and `.pdf` documents
- **Self-Reflection**: AI evaluates its own answer quality

## 📋 Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r pip-requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Gemini API key
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Prepare Your Data

```bash
# Create data directory
mkdir data

# Add your documents (.txt or .pdf files)
cp /path/to/your/documents/*.txt data/
cp /path/to/your/documents/*.pdf data/
```

### 4. Run the Application

```bash
# Launch Streamlit UI
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

## 📁 Project Structure

```
rag-agent/
├── rag_agent.py           # Core RAG agent with LangGraph workflow
├── streamlit_app.py       # Streamlit UI interface
├── evaluation.py          # Evaluation module with metrics
├── pip-requirements.txt   # Python dependencies
├── .env.example          # Example environment variables
├── .env                  # Your API keys (gitignored)
├── data/                 # Your documents directory
│   ├── document1.txt
│   └── document2.pdf
└── chroma_db/           # Vector database (auto-created)
```

## 🔧 Usage

### Using the Streamlit UI

1. **Configure**: Enter your Gemini API key in the sidebar
2. **Initialize**: Click "Initialize Agent" to load documents
3. **Query**: Type your question and click "Ask"
4. **View Results**: See answer, confidence scores, and reflection

### Using Python API

```python
from rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent(
    gemini_model="gemini-1.5-flash",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Load documents
agent.load_documents("./data")

# Build graph
agent.build_graph()

# Query
result = agent.query("What are the benefits of renewable energy?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Documents Retrieved: {result['num_docs_retrieved']}")
```

### Running Evaluations

```python
from rag_agent import RAGAgent
from evaluation import RAGEvaluator

# Initialize
agent = RAGAgent()
agent.load_documents("./data")
agent.build_graph()

evaluator = RAGEvaluator(agent=agent)

# Define test cases
test_cases = [
    {
        "question": "What are the benefits of renewable energy?",
        "reference_answer": "Renewable energy reduces emissions..."
    }
]

# Evaluate
results_df = evaluator.evaluate_dataset(test_cases)
evaluator.generate_report(results_df)
```

## 🏗️ Architecture

### LangGraph Workflow

```
┌─────────┐     ┌──────────┐     ┌────────┐     ┌─────────┐
│  PLAN   │────→│ RETRIEVE │────→│ ANSWER │────→│ REFLECT │
└─────────┘     └──────────┘     └────────┘     └─────────┘
    ↓               ↓                 ↓              ↓
Analyze      Find relevant    Generate       Evaluate
question     documents       response        quality
```

### Node Descriptions

1. **Plan Node**: Analyzes the query and determines if retrieval is needed
2. **Retrieve Node**: Searches vector database for relevant documents
3. **Answer Node**: Generates response using Gemini and retrieved context
4. **Reflect Node**: Evaluates answer quality and assigns confidence score

## 📊 Evaluation Metrics

The system supports multiple evaluation approaches:

### RAGAs-Style Metrics
- **Faithfulness**: Answer supported by retrieved context
- **Answer Relevancy**: Answer addresses the question
- **Context Relevancy**: Retrieved documents are relevant

### LLM-as-Judge (Gemini)
- **Accuracy**: Information correctness
- **Completeness**: Thoroughness of answer
- **Clarity**: Readability and structure
- **Helpfulness**: Usefulness to user

### Traditional NLP Metrics
- **ROUGE Scores**: N-gram overlap with reference
- **BERTScore**: Semantic similarity

## ⚙️ Configuration

### Model Options

**Gemini Models** (for LLM):
- `gemini-1.5-flash` (default, fast)
- `gemini-1.5-pro` (more capable)
- `gemini-pro` (legacy)

**Embedding Models** (HuggingFace):
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `sentence-transformers/all-mpnet-base-v2` (better quality)
- `BAAI/bge-small-en-v1.5` (multilingual)

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional - LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=rag-agent
```

## 🎯 Example Use Cases

- **Document Q&A**: Query your internal documentation
- **Research Assistant**: Ask questions about research papers
- **Knowledge Base**: Build a chatbot for your knowledge base
- **Customer Support**: Automate responses using your support docs

## 🐛 Troubleshooting

### API Key Issues
```
ValueError: GOOGLE_API_KEY not found
```
**Solution**: Make sure `.env` file exists with your API key

### No Documents Loaded
```
ValueError: No documents loaded!
```
**Solution**: Add `.txt` or `.pdf` files to the `data/` directory

### Import Errors
```
ModuleNotFoundError: No module named 'langchain_google_genai'
```
**Solution**: Run `pip install -r pip-requirements.txt`

### ChromaDB Errors
```
Solution: Delete the `chroma_db/` folder and reinitialize
```

## 📝 Development

### Running Tests

```bash
# Run evaluation on test cases
python evaluation.py
```

### Code Structure

- **State Management**: TypedDict for LangGraph state
- **Error Handling**: Comprehensive try-catch blocks
- **Logging**: Detailed console output for debugging
- **Modularity**: Separate files for agent, UI, and evaluation

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangChain & LangGraph**: Workflow orchestration
- **Google Gemini**: Advanced language model
- **Streamlit**: Beautiful UI framework
- **HuggingFace**: Embedding models

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the example code

## 🔗 Resources

- [Google Gemini API](https://ai.google.dev/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**Built with ❤️ using LangGraph, Gemini, and Streamlit**
