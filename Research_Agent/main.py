import streamlit as st
import os
from typing import List, Dict, Any
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated, TypedDict

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_history" not in st.session_state:
    st.session_state.research_history = []

# Configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ResearchReport(BaseModel):
    """Structure for research reports"""
    topic: str = Field(description="The research topic")
    key_findings: List[str] = Field(description="List of key findings")
    sources: List[str] = Field(description="List of sources used")
    summary: str = Field(description="Summary of the research")
    recommendations: List[str] = Field(description="Recommendations based on research")

@tool
def analyze_research_content(content: str, topic: str) -> Dict[str, Any]:
    """Analyze research content and extract key insights"""
    # This is a simplified analysis tool
    # In a real application, you might use more sophisticated NLP techniques
    
    lines = content.split('\n')
    key_points = [line.strip() for line in lines if line.strip() and len(line.strip()) > 50]
    
    return {
        "topic": topic,
        "content_length": len(content),
        "key_points": key_points[:5],  # Top 5 key points
        "analysis_timestamp": datetime.now().isoformat()
    }

# Define the state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_query: str
    search_results: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    final_report: str

class ResearchAssistantAgent:
    def _init(self, groq_api_key: str, tavily_api_key: str):  # Fixed: __init_ instead of init
        # Initialize the LLM with a current supported model
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",  # Updated to current supported model
            temperature=0.3
        )
        
        # Initialize tools
        self.search_tool = TavilySearchResults(
            api_wrapper_kwargs={
                "tavily_api_key": tavily_api_key,
                "search_depth": "advanced",
                "max_results": 5
            }
        )
        
        self.tools = [self.search_tool, analyze_research_content]
        
        # Create the agent workflow
        self.workflow = self._create_workflow()
        
    def _create_workflow(self):
        """Create the LangGraph workflow"""
        
        # Define the nodes
        def research_planner(state: AgentState):
            """Plan the research approach"""
            messages = state["messages"]
            system_message = SystemMessage(content="""
            You are a research planning assistant. Analyze the user's request and create a research plan.
            Break down complex topics into specific, searchable queries.
            Respond with a clear research strategy.
            """)
            
            response = self.llm.invoke([system_message] + messages)
            
            # Extract research query from the user's message
            user_message = next((msg.content for msg in messages if isinstance(msg, HumanMessage)), "")
            
            return {
                "messages": [response],
                "research_query": user_message
            }
        
        def web_searcher(state: AgentState):
            """Perform web search"""
            query = state["research_query"]
            
            try:
                search_results = self.search_tool.invoke(query)
                
                # Format search results
                formatted_results = []
                for result in search_results:
                    if isinstance(result, dict):
                        formatted_results.append({
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", "")
                        })
                
                search_summary = f"Found {len(formatted_results)} relevant sources for: {query}"
                
                return {
                    "messages": [AIMessage(content=search_summary)],
                    "search_results": formatted_results
                }
                
            except Exception as e:
                return {
                    "messages": [AIMessage(content=f"Search error: {str(e)}")],
                    "search_results": []
                }
        
        def content_analyzer(state: AgentState):
            """Analyze the search results"""
            search_results = state.get("search_results", [])
            query = state.get("research_query", "")
            
            if not search_results:
                return {
                    "messages": [AIMessage(content="No search results to analyze.")],
                    "analysis": {}
                }
            
            # Combine all content for analysis
            combined_content = "\n\n".join([
                f"Title: {result['title']}\nContent: {result['content']}"
                for result in search_results
            ])
            
            # Use the analysis tool
            analysis_result = analyze_research_content.invoke({
                "content": combined_content,
                "topic": query
            })
            
            return {
                "messages": [AIMessage(content="Content analysis completed.")],
                "analysis": analysis_result
            }
        
        def report_generator(state: AgentState):
            """Generate the final research report"""
            search_results = state.get("search_results", [])
            analysis = state.get("analysis", {})
            query = state.get("research_query", "")
            
            # Create a comprehensive prompt for report generation
            report_prompt = ChatPromptTemplate.from_template("""
            Based on the research query "{query}" and the following information, create a comprehensive research report:
            
            Search Results Summary:
            {search_summary}
            
            Analysis Results:
            {analysis}
            
            Please provide:
            1. Executive Summary
            2. Key Findings (with bullet points)
            3. Detailed Analysis
            4. Sources Used
            5. Recommendations
            
            Format the report in a clear, professional manner with proper sections.
            """)
            
            # Prepare the data for the prompt
            search_summary = "\n".join([
                f"• {result['title']}: {result['content'][:200]}..."
                for result in search_results[:3]  # Top 3 results
            ])
            
            analysis_text = json.dumps(analysis, indent=2)
            
            # Generate the report
            formatted_prompt = report_prompt.format(
                query=query,
                search_summary=search_summary,
                analysis=analysis_text
            )
            
            report_response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            return {
                "messages": [report_response],
                "final_report": report_response.content
            }
        
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", research_planner)
        workflow.add_node("searcher", web_searcher)
        workflow.add_node("analyzer", content_analyzer)
        workflow.add_node("reporter", report_generator)
        
        # Define the flow
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "searcher")
        workflow.add_edge("searcher", "analyzer")
        workflow.add_edge("analyzer", "reporter")
        workflow.add_edge("reporter", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def research(self, query: str) -> Dict[str, Any]:
        """Execute the research workflow"""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "research_query": query,
            "search_results": [],
            "analysis": {},
            "final_report": ""
        }
        
        config = {"configurable": {"thread_id": f"research_{datetime.now().timestamp()}"}}
        
        try:
            final_state = self.workflow.invoke(initial_state, config)
            return final_state
        except Exception as e:
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"Research failed: {str(e)}")],
                "final_report": f"Error occurred during research: {str(e)}"
            }

def validate_api_keys(groq_key: str, tavily_key: str) -> Dict[str, bool]:
    """Validate API keys by making test calls"""
    validation_results = {
        "groq_valid": False,
        "tavily_valid": False,
        "groq_error": "",
        "tavily_error": ""
    }
    
    # Test Groq API with current supported model
    try:
        test_llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.3-70b-versatile",  # Updated to current supported model
            temperature=0.1
        )
        # Simple test call
        test_response = test_llm.invoke([HumanMessage(content="Hi")])
        validation_results["groq_valid"] = True
    except Exception as e:
        validation_results["groq_error"] = str(e)
    
    # Test Tavily API
    try:
        test_search = TavilySearchResults(
            api_wrapper_kwargs={
                "tavily_api_key": tavily_key,
                "max_results": 1
            }
        )
        # Simple test search
        test_search.invoke("test")
        validation_results["tavily_valid"] = True
    except Exception as e:
        validation_results["tavily_error"] = str(e)
    
    return validation_results

def main():
    st.title("🔍 AI Research Assistant")
    st.markdown("Powered by LangChain + LangGraph")
    
    # Load API keys from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Check if API keys are provided
    if not groq_api_key or not tavily_api_key:
        st.error("❌ Missing API Keys!")
        st.markdown("""
        *Environment variables not found. Please set up your API keys:*
        
        1. **Create a .env file** in your project root with:
        
        GROQ_API_KEY=gsk_your_actual_groq_api_key_here
        TAVILY_API_KEY=tvly-your_actual_tavily_api_key_here
        
        
        2. *Get your API keys:*
        - Groq: [console.groq.com](https://console.groq.com) 
        - Tavily: [tavily.com](https://tavily.com)
        
        3. *Groq API Key Format:* Should start with gsk_
        4. *Tavily API Key Format:* Should start with tvly-
        
        5. *For deployment platforms* (Streamlit Cloud, Heroku, etc.), set these as environment variables in your platform's settings.
        """)
        st.stop()  # Stop execution here
    
    # Validate API key formats
    if not groq_api_key.startswith("gsk_"):
        st.error("❌ Invalid Groq API Key Format!")
        st.error("Groq API keys should start with 'gsk_'. Please check your .env file.")
        st.stop()
    
    if not tavily_api_key.startswith("tvly-"):
        st.error("❌ Invalid Tavily API Key Format!")
        st.error("Tavily API keys should start with 'tvly-'. Please check your .env file.")
        st.stop()
    
    # Test API keys (optional - can be slow, so make it toggleable)
    if "api_keys_validated" not in st.session_state:
        with st.spinner("🔍 Validating API keys..."):
            validation = validate_api_keys(groq_api_key, tavily_api_key)
            
            if not validation["groq_valid"]:
                st.error(f"❌ Groq API Key Invalid: {validation['groq_error']}")
                st.info("Please check your Groq API key at https://console.groq.com")
                st.stop()
            
            if not validation["tavily_valid"]:
                st.error(f"❌ Tavily API Key Invalid: {validation['tavily_error']}")
                st.info("Please check your Tavily API key at https://tavily.com")
                st.stop()
            
            st.session_state.api_keys_validated = True
            st.success("✅ All API keys validated successfully!")
    
    # Initialize the agent with error handling
    try:
        # Use a simple key since API keys are now from environment
        if "agent" not in st.session_state:
            with st.spinner("🚀 Initializing Research Assistant..."):
                st.session_state.agent = ResearchAssistantAgent(groq_api_key, tavily_api_key)
            st.success("✅ Research Assistant ready!")
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        st.error("Please check your API keys and internet connection.")
        st.info("Make sure your .env file contains valid API keys.")
        st.stop()
    
    # Research interface
    st.header("Research Query")
    
    # Research input
    research_query = st.text_area(
        "What would you like to research?",
        placeholder="Enter your research topic or question here...",
        height=100
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Start Research", type="primary"):
            if research_query.strip():
                with st.spinner("Conducting research... This may take a few moments."):
                    try:
                        result = st.session_state.agent.research(research_query)
                        
                        # Store in session state
                        st.session_state.research_history.append({
                            "query": research_query,
                            "result": result,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Research failed: {str(e)}")
            else:
                st.warning("Please enter a research query.")
    
    with col2:
        if st.button("🗑 Clear History"):
            st.session_state.research_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Display research results
    if st.session_state.research_history:
        st.header("Research Results")
        
        # Show latest research first
        for i, research in enumerate(reversed(st.session_state.research_history)):
            with st.expander(f"📊 Research: {research['query'][:50]}... ({research['timestamp']})", expanded=(i==0)):
                result = research['result']
                
                if 'error' in result:
                    st.error(f"Research failed: {result['error']}")
                else:
                    # Display the final report
                    if 'final_report' in result and result['final_report']:
                        st.markdown("### Research Report")
                        st.markdown(result['final_report'])
                    
                    # Display search results if available
                    if 'search_results' in result and result['search_results']:
                        st.markdown("### Sources")
                        for j, source in enumerate(result['search_results'], 1):
                            with st.container():
                                st.markdown(f"{j}. {source.get('title', 'Untitled')}")
                                if source.get('url'):
                                    st.markdown(f"🔗 [Source Link]({source.get('url')})")
                                if source.get('content'):
                                    st.markdown(f"{source.get('content')[:300]}...")
                                st.markdown("---")
    
    # Sidebar info
    st.sidebar.header("🤖 AI Research Assistant")
    st.sidebar.success("✅ API Keys Loaded from Environment")
    
    st.sidebar.header("About")
    st.sidebar.info("""
    This AI Research Assistant uses:
    - *LangGraph* for workflow orchestration
    - *LangChain* for LLM integration
    - *Groq* for fast inference
    - *Tavily* for web search
    
    The agent follows a structured workflow:
    1. 📋 Planning
    2. 🔍 Web Search
    3. 📊 Content Analysis
    4. 📄 Report Generation
    """)
    
    st.sidebar.header("API Keys Status")
    st.sidebar.success("🔑 Groq API: Loaded")
    st.sidebar.success("🔍 Tavily API: Loaded")
    
    if st.session_state.research_history:
        st.sidebar.metric("Completed Researches", len(st.session_state.research_history))

if _name_ == "_main":  # Fixed: __main_ instead of main
    main()
