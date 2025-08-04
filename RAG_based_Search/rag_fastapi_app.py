# This Python script implements a Retrieval-Augmented Generation (RAG) system
# and exposes it as a local API service using FastAPI.
#
# To run this app:
# 1. Install the required libraries:
#    pip install fastapi "uvicorn[standard]" pydantic langchain-community langchain-google-genai pypdf chromadb
# 2. Make sure your Google API key is set as an environment variable or enter it when prompted.
# 3. Place your downloaded PDF file in the specified path.
# 4. Run the script from your terminal: uvicorn rag_fastapi_app:app --reload

import os
import io
import requests
import getpass
import json
from typing import List, Optional

# Import FastAPI and Pydantic for API and structured data
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel, Field

# Import LangChain core components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Environment Setup (Executed once on server startup) ---
# Set your Google API key directly here for now.
# NOTE: It's recommended to use environment variables in a production setting.
GOOGLE_API_KEY = "API_HERE"

def setup_environment():
    """
    Sets up the environment by loading the Google API key from the script itself.
    """
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("Please replace 'YOUR_API_KEY_HERE' with your actual Google API key.")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Google API key loaded from script.")

# --- Define the structured output format using Pydantic v2 ---
class ClauseReference(PydanticBaseModel):
    doc_id: str = Field(description="The ID of the source document.")
    clause_id: str = Field(description="The specific clause or section number from the document.")
    text_excerpt: str = Field(description="A short excerpt from the referenced clause.")

class PolicyDecision(PydanticBaseModel):
    decision: str = Field(description="The final decision, e.g., 'Approved', 'Rejected', or 'Requires Further Review'.")
    amount: Optional[str] = Field(description="The payout amount, if applicable. Use 'N/A' if not.")
    justification: str = Field(description="A detailed explanation of the decision based on the policy clauses.")
    clause_references: List[ClauseReference] = Field(description="An array of references to the specific clauses used to make the decision.")

# --- Document Ingestion and Vector Store Creation (pre-loaded) ---
def ingest_document(pdf_path):
    """
    Loads a PDF from a local file path and splits it into chunks.
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Local file not found at: {pdf_path}")
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """
    Creates an in-memory vector store using ChromaDB and Google Generative AI embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store

# --- RAG Chain and Decision Engine ---
def create_rag_chain(vector_store):
    """
    Orchestrates the entire process: retrieval, generation, and output parsing.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Define the system prompt
    system_prompt = (
        "You are an expert insurance policy analyst. Your task is to analyze a claim query "
        "against the provided policy documents and make a decision. "
        "Follow these rules: "
        "- If the policy text mentions a waiting period for a procedure, check if the policy duration "
        "is less than that waiting period. If so, the decision is 'Rejected'. "
        "- If the policy states that a condition is not covered, the decision is 'Rejected'. "
        "- If the policy covers the procedure without any limiting clauses found, the decision is 'Approved'. "
        "- Always provide a clear justification based ONLY on the provided context. "
        "- If you cannot find relevant information, state that the decision requires further review."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context: {context}\n\nQuery: {query}")
    ])

    structured_llm = llm.with_structured_output(schema=PolicyDecision)
    llm_chain = prompt_template | structured_llm

    # A simple chain for direct retrieval without Pydantic parsing
    # The output from the LLM will be a raw Pydantic object
    return llm_chain, retriever

# --- FastAPI App Initialization and Endpoints ---
app = FastAPI()

# Global variables to store the initialized RAG components
rag_chain = None
retriever = None

@app.on_event("startup")
async def startup_event():
    global rag_chain, retriever
    
    # Path to your downloaded PDF file
    policy_pdf_source = "sample.pdf"
    
    try:
        setup_environment()
        print("Step 1: Ingesting and processing document...")
        chunks = ingest_document(policy_pdf_source)
        if not chunks:
            raise RuntimeError("Failed to process document. Exiting.")
        
        print("Step 2: Creating vector store...")
        vector_store = create_vector_store(chunks)
        
        print("Step 3: Creating RAG chain...")
        rag_chain, retriever = create_rag_chain(vector_store)
        
        print("RAG model API is ready.")
    except Exception as e:
        print(f"Failed to initialize RAG model: {e}")
        raise HTTPException(status_code=500, detail=f"Server startup failed: {e}")

# Define the request body for the API endpoint
class QueryRequest(PydanticBaseModel):
    query: str

@app.post("/api/v1/query")
async def process_query(request: QueryRequest):
    global rag_chain, retriever
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Model is not yet initialized.")
        
    try:
        # Use retriever.ainvoke for asynchronous calls
        retrieved_docs = await retriever.ainvoke(request.query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Invoke the chain to get a structured Pydantic object
        decision_object = await rag_chain.ainvoke({"context": context, "query": request.query})

        return {
            "decision": decision_object.decision,
            "amount": decision_object.amount,
            "justification": decision_object.justification,
            "clause_references": decision_object.clause_references
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
