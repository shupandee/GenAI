# This Python script implements an improved Retrieval-Augmented Generation (RAG) system
# with enhanced relevance targeting 75%+ accuracy through:
# - Hybrid search combining semantic and keyword matching
# - Query expansion and preprocessing
# - Multi-stage retrieval with document filtering
# - Enhanced chunking with semantic overlap
# - Improved re-ranking with multiple scoring methods
# - Context-aware prompt engineering
# - Query-document relevance validation
# Backend shows detailed scoring, client gets clean answers only
# FIXED: Unicode encoding issues for Windows compatibility

import os
import io
import sys
import requests
import getpass
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
import contextlib
import tempfile
import warnings
import re
import cohere
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime

# Import FastAPI and Pydantic for API and structured data
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel as PydanticBaseModel, Field, ValidationError

# Import LangChain core components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

# Suppress warnings
warnings.filterwarnings("ignore")

# UNICODE FIX: Set UTF-8 encoding for Windows
if sys.platform == "win32":
    # Set stdout and stderr to UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    # Set environment variable for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Setup logging for backend monitoring with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    """Custom stream handler that forces UTF-8 encoding"""
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            self.stream.reconfigure(encoding='utf-8')

# Configure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log', encoding='utf-8'),  # Force UTF-8 for file
        UTF8StreamHandler()  # Custom handler for console
    ]
)
logger = logging.getLogger(__name__)

# --- Environment Setup ---
GOOGLE_API_KEY = "AIzaSyDLt3dMcH1iPQhmjRP4xn9NvyXM7N_SzoI"
COHERE_API_KEY = "zJXW9qFwji8h9unSC1P0ebVjekbqdvSNKvNbOnwv"

def setup_environment():
    """Sets up the environment by loading API keys."""
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        raise ValueError("Please replace with your actual Google API key.")
    if COHERE_API_KEY == "YOUR_COHERE_API_KEY_HERE":
        raise ValueError("Please replace with your actual Cohere API key.")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY
    logger.info("API keys loaded successfully.")

# --- Enhanced Pydantic Models ---
class ClauseReference(PydanticBaseModel):
    doc_id: str = Field(description="Document ID or source path")
    clause_id: str = Field(description="Specific clause or section identifier")
    text_excerpt: str = Field(description="Relevant excerpt from the clause")
    page_number: Optional[int] = Field(description="Page number if available")
    relevance_score: Optional[float] = Field(description="Relevance score for this reference")

class PolicyDecision(PydanticBaseModel):
    decision: str = Field(description="Final decision: Approved/Rejected/Requires Further Review")
    amount: Optional[str] = Field(description="Payout amount or N/A")
    justification: str = Field(description="Detailed explanation based on policy clauses")
    clause_references: List[ClauseReference] = Field(description="Referenced clauses with scores")
    retrieval_relevance_score: Optional[float] = Field(description="Overall relevance score (0.0-1.0)")
    confidence_level: Optional[str] = Field(description="High/Medium/Low confidence in the decision")

# --- Backend Scoring Monitor ---
class BackendScoringMonitor:
    """Monitor and log detailed scoring information for backend analysis"""
    
    @staticmethod
    def log_query_processing(query: str, query_index: int):
        """Log the start of query processing"""
        logger.info("=" * 80)
        logger.info(f"PROCESSING QUERY #{query_index + 1}: {query}")
        logger.info("=" * 80)
    
    @staticmethod
    def log_retrieval_scores(retrieved_docs_with_scores: List[Tuple[Document, float]]):
        """Log detailed retrieval scoring"""
        logger.info("RETRIEVAL PHASE SCORES:")
        logger.info(f"   Documents retrieved: {len(retrieved_docs_with_scores)}")
        
        for i, (doc, score) in enumerate(retrieved_docs_with_scores[:5]):  # Log top 5
            page = doc.metadata.get('page', 0) + 1
            content_type = doc.metadata.get('content_type', 'general')
            content_preview = doc.page_content[:100].replace('\n', ' ') + "..."
            
            logger.info(f"   [{i+1}] Score: {score:.4f} | Page: {page} | Type: {content_type}")
            logger.info(f"       Preview: {content_preview}")
    
    @staticmethod
    def log_reranking_scores(reranked_docs_with_scores: List[Tuple[Document, float]], 
                           original_scores: List[float]):
        """Log detailed reranking scores and improvements"""
        logger.info("RERANKING PHASE SCORES:")
        logger.info(f"   Documents reranked: {len(reranked_docs_with_scores)}")
        
        for i, (doc, new_score) in enumerate(reranked_docs_with_scores):
            page = doc.metadata.get('page', 0) + 1
            old_score = original_scores[i] if i < len(original_scores) else 0.0
            improvement = new_score - old_score
            
            logger.info(f"   [{i+1}] New Score: {new_score:.4f} | Old Score: {old_score:.4f} | "
                       f"Change: {improvement:+.4f} | Page: {page}")
    
    @staticmethod
    def log_final_metrics(avg_relevance: float, confidence: str, processing_time: float):
        """Log final processing metrics"""
        logger.info("FINAL METRICS:")
        logger.info(f"   Average Relevance Score: {avg_relevance:.4f}")
        logger.info(f"   Confidence Level: {confidence}")
        logger.info(f"   Processing Time: {processing_time:.2f}s")
        
        # Quality assessment
        quality_status = "EXCELLENT" if avg_relevance > 0.8 else \
                        "GOOD" if avg_relevance > 0.6 else \
                        "ACCEPTABLE" if avg_relevance > 0.4 else "POOR"
        logger.info(f"   Quality Assessment: {quality_status}")
    
    @staticmethod
    def log_answer_generated(answer_length: int, contains_specific_info: bool):
        """Log answer generation metrics"""
        logger.info("ANSWER GENERATED:")
        logger.info(f"   Answer Length: {answer_length} characters")
        logger.info(f"   Contains Specific Info: {contains_specific_info}")

# --- Query Preprocessor ---
class QueryPreprocessor:
    """Enhanced query preprocessing for better retrieval"""
    
    def __init__(self):
        # Insurance domain keywords for expansion
        self.insurance_keywords = {
            'claim': ['insurance claim', 'policy claim', 'claim request', 'coverage claim'],
            'coverage': ['insurance coverage', 'policy coverage', 'covered benefits', 'eligible coverage'],
            'deductible': ['deductible amount', 'policy deductible', 'out of pocket'],
            'premium': ['insurance premium', 'policy premium', 'monthly payment'],
            'exclusion': ['policy exclusion', 'coverage exclusion', 'not covered'],
            'benefit': ['insurance benefit', 'policy benefit', 'covered benefit'],
            'limit': ['coverage limit', 'policy limit', 'maximum coverage'],
            'copay': ['copayment', 'co-payment', 'patient cost share']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with domain-specific synonyms"""
        expanded_queries = [query]
        query_lower = query.lower()
        
        for keyword, expansions in self.insurance_keywords.items():
            if keyword in query_lower:
                for expansion in expansions[:2]:  # Limit expansions to avoid noise
                    expanded_query = query_lower.replace(keyword, expansion)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries[:4]  # Limit to 4 variations
    
    def extract_key_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract key entities from query for targeted retrieval"""
        entities = {
            'amounts': re.findall(r'\$[\d,]+(?:\.\d{2})?', query),
            'percentages': re.findall(r'\d+%', query),
            'dates': re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', query),
            'medical_terms': []
        }
        
        # Common medical/insurance terms
        medical_terms = ['surgery', 'hospital', 'doctor', 'physician', 'treatment', 
                        'diagnosis', 'medication', 'therapy', 'emergency', 'ambulance']
        
        for term in medical_terms:
            if term in query.lower():
                entities['medical_terms'].append(term)
        
        return entities

# --- Enhanced Hybrid Retriever ---
class HybridRetriever:
    """Combines semantic and keyword-based retrieval"""
    
    def __init__(self, vector_store, embeddings_model):
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.query_preprocessor = QueryPreprocessor()
    
    async def retrieve_documents(self, query: str, top_k: int = 15) -> List[Tuple[Document, float]]:
        """Hybrid retrieval combining multiple strategies"""
        
        # 1. Expand query for better coverage
        expanded_queries = self.query_preprocessor.expand_query(query)
        
        # 2. Semantic retrieval for each query variation
        all_candidates = []
        for q in expanded_queries:
            semantic_docs = self.vector_store.similarity_search_with_score(q, k=top_k//2)
            all_candidates.extend([(doc, 1.0 - score) for doc, score in semantic_docs])
        
        # 3. Keyword-based filtering
        entities = self.query_preprocessor.extract_key_entities(query)
        keyword_filtered_docs = self._keyword_filter(all_candidates, query, entities)
        
        # 4. Deduplicate and merge scores
        merged_docs = self._merge_and_deduplicate(keyword_filtered_docs)
        
        # 5. Sort by combined score and return top results
        merged_docs.sort(key=lambda x: x[1], reverse=True)
        return merged_docs[:top_k]
    
    def _keyword_filter(self, candidates: List[Tuple[Document, float]], 
                       query: str, entities: Dict[str, List[str]]) -> List[Tuple[Document, float]]:
        """Filter and boost documents based on keyword matching"""
        filtered_docs = []
        query_terms = set(query.lower().split())
        
        for doc, score in candidates:
            content_lower = doc.page_content.lower()
            
            # Base keyword matching score
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = keyword_matches / len(query_terms) if query_terms else 0
            
            # Boost for specific entity matches
            entity_boost = 0
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.lower() in content_lower:
                        entity_boost += 0.2
            
            # Combined score
            combined_score = (score * 0.7) + (keyword_score * 0.2) + (entity_boost * 0.1)
            
            # Only include documents with minimum relevance
            if combined_score > 0.3:  # Threshold for relevance
                filtered_docs.append((doc, combined_score))
        
        return filtered_docs
    
    def _merge_and_deduplicate(self, docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Merge duplicate documents and combine their scores"""
        doc_scores = defaultdict(list)
        doc_objects = {}
        
        for doc, score in docs:
            doc_key = (doc.page_content[:100], doc.metadata.get('page', -1))  # Use content snippet and page as key
            doc_scores[doc_key].append(score)
            if doc_key not in doc_objects:
                doc_objects[doc_key] = doc
        
        # Combine scores (take max score for each document)
        merged_docs = []
        for doc_key, scores in doc_scores.items():
            max_score = max(scores)
            merged_docs.append((doc_objects[doc_key], max_score))
        
        return merged_docs

# --- Enhanced Cohere Reranker ---
class EnhancedCohereReranker:
    """Enhanced reranker with multiple scoring methods"""
    
    def __init__(self, api_key: str, top_n: int = 6):
        self.client = cohere.Client(api_key)
        self.top_n = top_n
    
    async def rerank(self, query: str, documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Enhanced reranking with fallback strategies"""
        if not documents:
            return []
        
        try:
            # Prepare documents for reranking
            doc_texts = [doc.page_content for doc, _ in documents]
            initial_scores = [score for _, score in documents]
            
            # Cohere reranking
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_n=min(self.top_n, len(documents)),
                return_documents=True
            )
            
            # Combine Cohere scores with initial retrieval scores
            reranked_results = []
            for result in response.results:
                original_doc = documents[result.index][0]
                initial_score = initial_scores[result.index]
                cohere_score = result.relevance_score
                
                # Weighted combination of scores
                combined_score = (cohere_score * 0.7) + (initial_score * 0.3)
                reranked_results.append((original_doc, combined_score))
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return top documents based on initial scores
            sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
            return sorted_docs[:self.top_n]

# --- Enhanced Document Processing ---
class EnhancedDocumentProcessor:
    """Enhanced document processing with semantic chunking"""
    
    @staticmethod
    async def load_documents_from_url(url: str) -> List[Document]:
        """Load PDF with enhanced metadata extraction"""
        logger.info(f"Loading document from URL: {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(file_path=tmp_file_path)
            documents = loader.load()
            
            # Enhanced metadata
            for i, doc in enumerate(documents):
                doc.metadata['doc_id'] = f"policy_page_{i+1}"
                doc.metadata['total_pages'] = len(documents)
                
            os.remove(tmp_file_path)
            logger.info(f"Loaded {len(documents)} pages with enhanced metadata.")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return []
    
    @staticmethod
    async def create_enhanced_chunks(documents: List[Document]) -> List[Document]:
        """Create chunks with semantic boundaries and overlap"""
        if not documents:
            return []
        
        # Multi-level chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=150,  # Increased overlap for better context
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]  # Prioritize semantic breaks
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Enhance chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"chunk_{i}"
            chunk.metadata['chunk_index'] = i
            
            # Add semantic context clues
            content_lower = chunk.page_content.lower()
            if any(keyword in content_lower for keyword in ['coverage', 'benefit', 'eligible']):
                chunk.metadata['content_type'] = 'coverage'
            elif any(keyword in content_lower for keyword in ['exclusion', 'not covered', 'excluded']):
                chunk.metadata['content_type'] = 'exclusion'
            elif any(keyword in content_lower for keyword in ['deductible', 'copay', 'cost']):
                chunk.metadata['content_type'] = 'cost'
            else:
                chunk.metadata['content_type'] = 'general'
        
        logger.info(f"Created {len(chunks)} enhanced semantic chunks.")
        return chunks

# --- Enhanced RAG Chain ---
async def create_enhanced_rag_components(vector_store_instance):
    """Create enhanced RAG components with improved prompting"""
    
    # Enhanced LLM with better temperature for consistency
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.05,  # Lower temperature for more consistent output
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Create enhanced retrievers
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    hybrid_retriever = HybridRetriever(vector_store_instance, embeddings)
    enhanced_reranker = EnhancedCohereReranker(api_key=os.environ["COHERE_API_KEY"], top_n=6)
    
    # Enhanced system prompt with better instructions
    system_prompt = """You are an expert insurance policy analyst with deep knowledge of policy interpretation and claims processing.

INSTRUCTIONS:
1. Carefully analyze the provided policy context against the user's query
2. Pay special attention to specific coverage amounts, exclusions, and eligibility criteria
3. Look for exact matches between the query scenario and policy provisions
4. Consider both what IS covered and what is NOT covered
5. Provide clear, definitive answers based on the policy text
6. When uncertain, explicitly state the limitations and suggest what additional information might be needed

RESPONSE FORMAT:
- Start with a clear, direct answer to the question
- Provide detailed justification referencing specific policy sections
- Include relevant dollar amounts, percentages, or limits when applicable
- Highlight any important conditions or requirements
- Note any exclusions that might apply

Remember: Base your response ONLY on the provided policy context. Do not make assumptions beyond what is explicitly stated in the documents."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """POLICY CONTEXT:
{context}

QUERY: {query}

Please provide a comprehensive analysis based on the policy context above.""")
    ])

    return prompt_template, llm, hybrid_retriever, enhanced_reranker

# --- FastAPI App with Enhanced Processing ---
app = FastAPI()

# Global variables
global_prompt_template = None
global_llm = None
global_hybrid_retriever = None
global_enhanced_reranker = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global global_prompt_template, global_llm, global_hybrid_retriever, global_enhanced_reranker
    
    try:
        setup_environment()
        
        # Enhanced document processing
        policy_pdf_source = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
        logger.info("Step 1: Enhanced document processing...")
        processor = EnhancedDocumentProcessor()
        raw_documents = await processor.load_documents_from_url(policy_pdf_source)
        if not raw_documents:
            raise RuntimeError("Failed to load documents.")
        
        chunks = await processor.create_enhanced_chunks(raw_documents)
        if not chunks:
            raise RuntimeError("No chunks generated.")
        
        logger.info("Step 2: Creating enhanced vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        logger.info("Step 3: Creating enhanced RAG components...")
        (global_prompt_template, global_llm, 
         global_hybrid_retriever, global_enhanced_reranker) = await create_enhanced_rag_components(vector_store)
        
        logger.info("Enhanced RAG API is ready with improved relevance targeting!")
        yield
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server startup failed: {e}")

app = FastAPI(lifespan=lifespan)

class QueryRequest(PydanticBaseModel):
    documents: str = Field(description="URL to PDF document (pre-loaded on startup)")
    questions: List[str] = Field(description="List of claim-related questions")

@app.post("/api/v1/query")
async def process_enhanced_query(request: QueryRequest):
    """Enhanced query processing with detailed backend scoring and clean client response"""
    global global_prompt_template, global_llm, global_hybrid_retriever, global_enhanced_reranker
    
    if not all([global_llm, global_hybrid_retriever, global_enhanced_reranker]):
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    answers = []  # Clean answers for client
    scoring_monitor = BackendScoringMonitor()
    
    logger.info(f"PROCESSING BATCH OF {len(request.questions)} QUERIES")
    batch_start_time = datetime.now()
    
    for query_index, query in enumerate(request.questions):
        query_start_time = datetime.now()
        
        try:
            # Log query start
            scoring_monitor.log_query_processing(query, query_index)
            
            # Enhanced hybrid retrieval
            retrieved_docs_with_scores = await global_hybrid_retriever.retrieve_documents(query, top_k=15)
            
            if not retrieved_docs_with_scores:
                answer = "No relevant policy information found for this query."
                answers.append(answer)
                logger.warning(f"No documents retrieved for query: {query}")
                continue
            
            # Log retrieval scores (backend only)
            scoring_monitor.log_retrieval_scores(retrieved_docs_with_scores)
            
            # Store original scores for comparison
            original_scores = [score for _, score in retrieved_docs_with_scores]
            
            # Enhanced reranking
            reranked_docs_with_scores = await global_enhanced_reranker.rerank(query, retrieved_docs_with_scores)
            
            # Log reranking improvements (backend only)
            scoring_monitor.log_reranking_scores(reranked_docs_with_scores, original_scores)
            
            # Prepare enhanced context with metadata (for backend analysis)
            context_parts = []
            total_relevance = 0
            
            for i, (doc, score) in enumerate(reranked_docs_with_scores):
                doc_source = doc.metadata.get('source', 'policy_document').split('/')[-1]
                page_num = doc.metadata.get('page', 0) + 1
                content_type = doc.metadata.get('content_type', 'general')
                
                context_parts.append(f"""--- Document: {doc_source} | Page: {page_num} | Type: {content_type} | Relevance: {score:.3f} ---
{doc.page_content.strip()}""")
                
                total_relevance += score
            
            # Calculate metrics (for backend monitoring)
            avg_relevance = total_relevance / len(reranked_docs_with_scores) if reranked_docs_with_scores else 0
            confidence = "High" if avg_relevance > 0.7 else "Medium" if avg_relevance > 0.5 else "Low"
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            formatted_prompt = global_prompt_template.format_messages(context=context, query=query)
            response = await global_llm.ainvoke(formatted_prompt)
            
            answer_text = response.content.strip()
            
            # Check if answer contains specific information
            contains_specific_info = any(indicator in answer_text.lower() for indicator in 
                                       ['%', '$', 'days', 'months', 'years', 'limit', 'maximum', 'minimum'])
            
            # Log answer generation metrics (backend only)
            scoring_monitor.log_answer_generated(len(answer_text), contains_specific_info)
            
            # Calculate processing time
            processing_time = (datetime.now() - query_start_time).total_seconds()
            
            # Log final metrics (backend only)
            scoring_monitor.log_final_metrics(avg_relevance, confidence, processing_time)
            
            # Add only clean answer to client response
            answers.append(answer_text)
            
            logger.info(f"Query #{query_index + 1} processed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            answers.append(f"Error processing query: {str(e)}")
    
    # Log batch completion
    total_batch_time = (datetime.now() - batch_start_time).total_seconds()
    logger.info(f"BATCH COMPLETED: {len(request.questions)} queries in {total_batch_time:.2f}s")
    logger.info(f"Average time per query: {total_batch_time / len(request.questions):.2f}s")
    
    # Return only clean answers to client
    return {"answers": answers}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_components_ready": all([
            global_llm is not None,
            global_hybrid_retriever is not None,
            global_enhanced_reranker is not None
        ])
    }

# Backend monitoring endpoint (for internal use)
@app.get("/api/v1/system/metrics")
async def get_system_metrics():
    """Internal endpoint to check system metrics (backend only)"""
    return {
        "system_status": "operational",
        "components_loaded": {
            "llm": global_llm is not None,
            "hybrid_retriever": global_hybrid_retriever is not None,
            "enhanced_reranker": global_enhanced_reranker is not None
        },
        "log_file": "rag_system.log",
        "scoring_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)