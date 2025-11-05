"""
RAG-based Q&A AI Agent using LangGraph and Gemini
Uses Hugging Face for embeddings only, Gemini for LLM
Loads API keys from .env file
"""

import os
from typing import TypedDict, List, Annotated
from datetime import datetime
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State object that flows through the graph"""
    question: str
    retrieval_needed: bool
    retrieved_docs: List[Document]
    answer: str
    reflection: str
    is_relevant: bool
    confidence_score: float
    timestamp: str
    steps_log: Annotated[List[str], operator.add]


# ============================================================================
# RAG AGENT CLASS
# ============================================================================

class RAGAgent:
    """LangGraph-based RAG Agent with 4-node workflow"""
    
    def __init__(self, 
                 gemini_api_key: str = None,
                 gemini_model: str = "gemini-2.5-flash",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG Agent
        
        Args:
            gemini_api_key: Google Gemini API key (if None, reads from env)
            gemini_model: Gemini model name (gemini-1.5-flash or gemini-1.5-pro)
            embedding_model: HuggingFace model for embeddings
            persist_directory: Directory to persist ChromaDB
        """
        # Load API key from environment if not provided
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in .env file or pass as parameter.")
        
        self.persist_directory = persist_directory
        
        print(f"[INIT] Initializing RAG Agent...")
        print(f"[INIT] LLM Model: Google {gemini_model}")
        print(f"[INIT] Embedding Model: {embedding_model}")
        
        # Initialize embeddings (Hugging Face - no API key needed for sentence-transformers)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=self.gemini_api_key,
            temperature=0.7,
            max_output_tokens=512,
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.graph = None
        
        print("[INIT] ✓ Agent initialized successfully\n")
    
    def load_documents(self, data_path: str):
        """Load and process documents into vector store"""
        print(f"[LOAD] Loading documents from: {data_path}")
        
        documents = []
        
        # Load .txt files
        try:
            txt_loader = DirectoryLoader(
                data_path, 
                glob="**/*.txt", 
                loader_cls=TextLoader,
                show_progress=True
            )
            documents.extend(txt_loader.load())
            print(f"[LOAD] Loaded {len(documents)} text documents")
        except Exception as e:
            print(f"[LOAD] No .txt files found or error: {e}")
        
        # Load .pdf files
        try:
            pdf_loader = DirectoryLoader(
                data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"[LOAD] Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"[LOAD] No .pdf files found or error: {e}")
        
        if not documents:
            raise ValueError("No documents loaded! Please check your data path.")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        print(f"[LOAD] Split into {len(splits)} chunks")
        
        # Create vector store
        print("[LOAD] Creating vector embeddings...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"[LOAD] ✓ Vector store created with {len(splits)} embeddings\n")
    
    # ========================================================================
    # NODE 1: PLAN
    # ========================================================================
    
    def plan_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Analyze the question and decide if retrieval is needed
        """
        print("\n" + "="*70)
        print("NODE 1: PLAN - Analyzing query")
        print("="*70)
        
        question = state["question"]
        print(f"[PLAN] Question: {question}")
        
        # Simple heuristic: Check if question requires factual information
        retrieval_keywords = [
            "what", "how", "why", "when", "where", "who",
            "explain", "describe", "tell me", "benefits", "advantages",
            "disadvantages", "define", "list", "compare"
        ]
        
        question_lower = question.lower()
        needs_retrieval = any(kw in question_lower for kw in retrieval_keywords)
        
        # Advanced check using Gemini
        try:
            plan_prompt = f"""Question: {question}

Does this question require retrieving information from a knowledge base? Answer only YES or NO."""
            
            response = self.llm.invoke(plan_prompt)
            llm_decision = response.content if hasattr(response, 'content') else str(response)
            needs_retrieval = "yes" in llm_decision.lower()[:50]
            print(f"[PLAN] Gemini Decision: {llm_decision[:200]}...")
        except Exception as e:
            print(f"[PLAN] Gemini call failed, using heuristic: {e}")
            # Fall back to heuristic-based decision
        
        state["retrieval_needed"] = needs_retrieval
        state["steps_log"].append(f"PLAN: Retrieval needed = {needs_retrieval}")
        
        print(f"[PLAN] ✓ Decision: {'Retrieval NEEDED' if needs_retrieval else 'Direct answer possible'}")
        
        return state
    
    # ========================================================================
    # NODE 2: RETRIEVE
    # ========================================================================
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Retrieve relevant documents from vector store
        """
        print("\n" + "="*70)
        print("NODE 2: RETRIEVE - Fetching relevant documents")
        print("="*70)
        
        if not state["retrieval_needed"]:
            print("[RETRIEVE] ⊘ Skipping retrieval (not needed)")
            state["retrieved_docs"] = []
            state["steps_log"].append("RETRIEVE: Skipped")
            return state
        
        question = state["question"]
        print(f"[RETRIEVE] Searching for: {question}")
        
        # Use invoke() instead of deprecated get_relevant_documents()
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        docs = retriever.invoke(question)
        state["retrieved_docs"] = docs
        
        print(f"[RETRIEVE] ✓ Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs, 1):
            print(f"\n[RETRIEVE] Document {i}:")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content Preview: {doc.page_content[:150]}...")
        
        state["steps_log"].append(f"RETRIEVE: Found {len(docs)} documents")
        
        return state
    
    # ========================================================================
    # NODE 3: ANSWER
    # ========================================================================
    
    def answer_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Generate answer using Gemini and retrieved context
        """
        print("\n" + "="*70)
        print("NODE 3: ANSWER - Generating response")
        print("="*70)
        
        question = state["question"]
        docs = state["retrieved_docs"]
        
        if docs:
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            formatted_prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
            
        else:
            # No retrieval, direct answer
            formatted_prompt = f"""Answer the following question clearly and concisely:

Question: {question}

Answer:"""
        
        print("[ANSWER] Querying Gemini...")
        
        try:
            response = self.llm.invoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            answer = answer.strip()
        except Exception as e:
            print(f"[ANSWER] Error generating answer: {e}")
            answer = "I apologize, but I encountered an error generating an answer. Please try rephrasing your question."
        
        state["answer"] = answer
        state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["steps_log"].append(f"ANSWER: Generated ({len(answer)} chars)")
        
        print(f"[ANSWER] ✓ Generated answer ({len(answer)} characters)")
        print(f"\n[ANSWER] Response Preview:\n{answer[:300]}...\n")
        
        return state
    
    # ========================================================================
    # NODE 4: REFLECT
    # ========================================================================
    
    def reflect_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Evaluate answer for relevance and quality using Gemini
        """
        print("\n" + "="*70)
        print("NODE 4: REFLECT - Evaluating answer quality")
        print("="*70)
        
        question = state["question"]
        answer = state["answer"]
        
        reflection_prompt = f"""Evaluate this Q&A pair:

Question: {question}
Answer: {answer}

Provide:
1. A relevance score from 0-10
2. Brief explanation of quality

Format: Score: X/10
Explanation: [your assessment]"""
        
        print("[REFLECT] Analyzing answer quality with Gemini...")
        
        try:
            response = self.llm.invoke(reflection_prompt)
            reflection = response.content if hasattr(response, 'content') else str(response)
            state["reflection"] = reflection
            
            # Extract confidence score
            reflection_lower = reflection.lower()
            
            # Try to extract numeric score
            import re
            score_match = re.search(r'(\d+)/10', reflection)
            if score_match:
                score = int(score_match.group(1))
                confidence = score / 10.0
            elif any(word in reflection_lower for word in ["excellent", "good", "relevant", "accurate"]):
                confidence = 0.8
                score = 8
            elif any(word in reflection_lower for word in ["poor", "irrelevant", "unclear", "inadequate"]):
                confidence = 0.3
                score = 3
            else:
                confidence = 0.6
                score = 6
            
            is_relevant = score >= 5
            
            state["confidence_score"] = confidence
            state["is_relevant"] = is_relevant
            
            print(f"[REFLECT] ✓ Evaluation complete")
            print(f"[REFLECT] Confidence Score: {confidence:.2f}")
            print(f"[REFLECT] Is Relevant: {is_relevant}")
            print(f"\n[REFLECT] Reflection:\n{reflection[:300]}...\n")
            
        except Exception as e:
            print(f"[REFLECT] Error during reflection: {e}")
            state["reflection"] = "Reflection unavailable"
            state["confidence_score"] = 0.5
            state["is_relevant"] = True
        
        state["steps_log"].append(f"REFLECT: Score={state['confidence_score']:.2f}")
        
        return state
    
    # ========================================================================
    # GRAPH CONSTRUCTION
    # ========================================================================
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        print("[GRAPH] Building LangGraph workflow...")
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("answer", self.answer_node)
        workflow.add_node("reflect", self.reflect_node)
        
        # Define edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", "reflect")
        workflow.add_edge("reflect", END)
        
        # Compile graph
        self.graph = workflow.compile()
        
        print("[GRAPH] ✓ Graph compiled successfully\n")
    
    def query(self, question: str) -> dict:
        """
        Query the agent with a question
        
        Args:
            question: User's question
            
        Returns:
            dict containing answer and metadata
        """
        if not self.graph:
            self.build_graph()
        
        print("\n" + "█"*70)
        print("NEW QUERY SESSION")
        print("█"*70)
        
        # Initialize state
        initial_state = AgentState(
            question=question,
            retrieval_needed=False,
            retrieved_docs=[],
            answer="",
            reflection="",
            is_relevant=False,
            confidence_score=0.0,
            timestamp="",
            steps_log=[]
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "█"*70)
        print("QUERY COMPLETE")
        print("█"*70)
        print(f"\n[SUMMARY] Workflow Steps:")
        for step in final_state["steps_log"]:
            print(f"  • {step}")
        print()
        
        return {
            "question": final_state["question"],
            "answer": final_state["answer"],
            "confidence": final_state["confidence_score"],
            "is_relevant": final_state["is_relevant"],
            "reflection": final_state["reflection"],
            "num_docs_retrieved": len(final_state["retrieved_docs"]),
            "timestamp": final_state["timestamp"],
            "retrieved_docs": final_state["retrieved_docs"]
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # API key loaded from .env automatically
    agent = RAGAgent(
        gemini_model="gemini-2.5-flash",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load documents (create a 'data' folder with .txt or .pdf files)
    agent.load_documents("./data")
    
    # Build the graph
    agent.build_graph()
    
    # Example queries
    questions = [
        "What are the benefits of renewable energy?",
        "How does solar power work?",
        "What are the main challenges in renewable energy adoption?"
    ]
    
    for question in questions:
        result = agent.query(question)
        print(f"\n{'='*70}")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"{'='*70}\n")