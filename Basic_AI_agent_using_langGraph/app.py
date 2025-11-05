"""
Streamlit UI for RAG AI Agent with Gemini
Interactive interface for querying the knowledge base
"""

import streamlit as st
import os
from rag_agent import RAGAgent
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .workflow-step {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.3rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_agent(gemini_api_key, gemini_model, embedding_model, data_path):
    """Initialize the RAG agent"""
    try:
        with st.spinner("🔄 Initializing RAG Agent..."):
            # Create agent
            agent = RAGAgent(
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
                embedding_model=embedding_model
            )
            
            # Load documents
            agent.load_documents(data_path)
            
            # Build graph
            agent.build_graph()
            
            st.session_state.agent = agent
            st.session_state.initialized = True
            
        return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False


def main():
    # Header
    st.markdown("<h1 class='main-header'>🤖 RAG Q&A AI Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Powered by Google Gemini & LangGraph</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Token
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        # Model selection
        st.subheader("Model Settings")
        
        gemini_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-pro"
            ],
            help="Gemini model for answer generation and reflection"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5"
            ],
            help="HuggingFace model for document embeddings (runs locally)"
        )
        
        # Data path
        data_path = st.text_input(
            "Data Directory",
            value="./data",
            help="Path to folder containing .txt or .pdf files"
        )
        
        st.markdown("---")
        
        # Initialize button
        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("⚠️ Please provide a Google Gemini API key")
            elif not os.path.exists(data_path):
                st.error(f"⚠️ Data directory not found: {data_path}")
            else:
                if initialize_agent(gemini_api_key, gemini_model, embedding_model, data_path):
                    st.success("✅ Agent initialized successfully!")
                    st.balloons()
        
        # Status
        st.markdown("---")
        st.subheader("📊 Status")
        if st.session_state.initialized:
            st.success("✅ Agent Ready")
            st.info(f"📚 Queries: {len(st.session_state.history)}")
        else:
            st.warning("⏳ Not Initialized")
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        # Info section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **RAG Workflow:**
        1. 📋 **Plan** - Analyze query
        2. 🔍 **Retrieve** - Find relevant docs
        3. 💬 **Answer** - Generate response
        4. 🎯 **Reflect** - Evaluate quality
        """)
    
    # Main content
    if not st.session_state.initialized:
        st.info("👈 Please configure and initialize the agent in the sidebar to get started.")
        
        # Show example
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📖 How to Use")
            st.markdown("""
            1. **Get a Google Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. **Prepare your data**: Create a `data` folder with `.txt` or `.pdf` files
            3. **Configure settings** in the sidebar
            4. **Initialize the agent**
            5. **Ask questions** about your documents!
            """)
        
        with col2:
            st.subheader("💡 Example Questions")
            st.code("""
• What are the benefits of renewable energy?
• How does solar power work?
• Explain the challenges in wind energy
• Compare different types of renewable sources
            """, language=None)
        
        st.subheader("✨ Features")
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        with feature_col1:
            st.markdown("### 📋 Plan")
            st.write("Intelligent query analysis")
        
        with feature_col2:
            st.markdown("### 🔍 Retrieve")
            st.write("Semantic document search")
        
        with feature_col3:
            st.markdown("### 💬 Answer")
            st.write("Context-aware responses")
        
        with feature_col4:
            st.markdown("### 🎯 Reflect")
            st.write("Quality evaluation")
        
    else:
        # Query interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("💬 Ask a Question")
            
        with col2:
            if st.button("📋 Example Questions", use_container_width=True):
                st.session_state.show_examples = not st.session_state.get('show_examples', False)
        
        # Show examples if toggled
        if st.session_state.get('show_examples', False):
            with st.expander("💡 Example Questions", expanded=True):
                example_questions = [
                    "What are the main types of renewable energy?",
                    "How does solar energy benefit the environment?",
                    "What are the challenges in renewable energy adoption?",
                    "Compare wind and solar energy advantages",
                    "Explain how hydroelectric power works"
                ]
                for eq in example_questions:
                    if st.button(eq, key=f"ex_{eq}"):
                        st.session_state.current_question = eq
                        st.rerun()
        
        # Question input
        question = st.text_input(
            "Your Question:",
            value=st.session_state.get('current_question', ''),
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit = st.button("🔍 Ask", type="primary", use_container_width=True)
        with col2:
            clear_question = st.button("🔄 Clear", use_container_width=True)
        
        if clear_question:
            st.session_state.current_question = ''
            st.rerun()
        
        # Process query
        if submit and question:
            with st.spinner("🤔 Processing through RAG workflow..."):
                start_time = time.time()
                
                try:
                    result = st.session_state.agent.query(question)
                    end_time = time.time()
                    
                    result['processing_time'] = end_time - start_time
                    
                    # Add to history
                    st.session_state.history.insert(0, result)
                    
                    # Clear current question
                    st.session_state.current_question = ''
                    
                    st.success("✅ Answer generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display results
        if st.session_state.history:
            st.markdown("---")
            st.subheader("📋 Results")
            
            for idx, result in enumerate(st.session_state.history):
                with st.container():
                    # Question
                    st.markdown(f"### ❓ Question {len(st.session_state.history) - idx}")
                    st.info(result['question'])
                    
                    # Answer
                    st.markdown("### 💡 Answer")
                    st.success(result['answer'])
                    
                    # Metadata
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.2%}",
                            delta=None
                        )
                    
                    with col2:
                        relevance_display = "✅ Relevant" if result['is_relevant'] else "❌ Not Relevant"
                        st.metric(
                            "Relevance",
                            relevance_display
                        )
                    
                    with col3:
                        st.metric(
                            "Documents",
                            result['num_docs_retrieved']
                        )
                    
                    with col4:
                        st.metric(
                            "Time",
                            f"{result.get('processing_time', 0):.2f}s"
                        )
                    
                    # Retrieved Documents (if available)
                    if result.get('retrieved_docs') and len(result['retrieved_docs']) > 0:
                        with st.expander("📄 Retrieved Documents"):
                            for i, doc in enumerate(result['retrieved_docs'], 1):
                                st.markdown(f"**Document {i}:**")
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.markdown("---")
                    
                    # Reflection (collapsible)
                    with st.expander("🔍 AI Evaluation & Reflection"):
                        st.markdown("**Quality Assessment:**")
                        st.text(result['reflection'])
                        st.caption(f"Timestamp: {result['timestamp']}")
                    
                    st.markdown("---")


if __name__ == "__main__":
    main()